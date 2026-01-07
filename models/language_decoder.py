import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Utility Functions (Rotery Positional Embeddings - Q & K)

# a: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# b: Reshape frequency tensor for broadcasting it with another tensor.
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# c: Apply rotary embeddings to input tensors using the given frequency tensor.
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Grouped Query Attention

# a: repeat kv for specified intervals
def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# b: Attention layer
class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, config):
        super().__init__()
        self.n_kv_heads = config.num_kv_heads
        self.n_heads = config.num_heads
        self.n_rep = config.num_heads // config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.wq = nn.Linear(
            config.embed_dim,
            config.num_heads * self.head_dim,
            bias=False,
        )
        
        self.wk = nn.Linear(
            config.embed_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        
        self.wv = nn.Linear(
            config.embed_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        
        self.wo = nn.Linear(
            config.num_heads * self.head_dim,
            config.embed_dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (
                config.batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                config.batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

    def forward(self,x,start_pos,freqs_cis,mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq.device)
        self.cache_v = self.cache_v.to(xq.device)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)
        
        # scaled dot product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# c: Cross-Attention Layer
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections of query, key, value
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # Apply attention weights to the values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate the heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear layer
        output = self.out_linear(output)  # [batch_size, seq_len, embed_dim]

        return output, attention_weights

# FeedForward SwiGLU
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim,):
        super().__init__()

        self.w1 = nn.Linear(
            embed_dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, embed_dim, bias=False,
        )
        self.w3 = nn.Linear(
            embed_dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))        

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.n_heads = config.num_heads
        self.dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        
        self.self_attention = Attention(config)

        self.cross_attention = MultiheadAttention(config.embed_dim, config.num_heads) 
    
        self.feed_forward = FeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
        )
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

    def forward(self, lang_seq, img_features, start_pos, freqs_cis, mask,):
        h1 = lang_seq + self.self_attention(
            self.attention_norm(lang_seq), start_pos, freqs_cis, mask
        )

        h2, attn_weights = self.cross_attention(
            self.attention_norm(h1), self.attention_norm(img_features), self.attention_norm(img_features)
        )
        
        h2 = h1 + h2
        out = h2 + self.feed_forward(self.ffn_norm(h2))
        
        return out

# LLaMA-X Decoder
class LLaMAXDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_layers
        self.head_dim = config.embed_dim // config.num_heads

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.embed_dim,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.num_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.output = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.head_dim, self.config.max_seq_len * 2
        )

    def forward(self, tokens, img_embs, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, img_embs, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
