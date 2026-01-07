import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.max_seq_length = config.max_seq_len

        self.logger = config.get_logger(__name__)
        self.logger.info(f"Initializing Transformer Language Encoder with {self.num_layers} layers.")
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(self.max_seq_length, self.embed_dim), requires_grad=False)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
    
    def _generate_positional_encoding(self, max_seq_length, embed_dim):
        """
        Generate sinusoidal positional encoding.
        """
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): Input tensor of shape (batch_size, seq_length), containing token indices.
            attention_mask (Tensor, optional): Mask tensor of shape (batch_size, seq_length), where 1 indicates
                                               tokens to be attended to and 0 indicates tokens to be ignored.
        Returns:
            Tensor: Output embeddings of shape (batch_size, seq_length, embed_dim).
        """
        batch_size, seq_length = input_ids.shape
        
        # Embedding lookup
        embeddings = self.embedding(input_ids)  # (batch_size, seq_length, embed_dim)
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:, :seq_length, :]
        
        # Prepare mask for transformer encoder
        if attention_mask is not None:
            # Convert attention mask to the format required by nn.TransformerEncoder
            # Shape should be (seq_length, seq_length) per nn.TransformerEncoder requirements
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)  # (batch_size, seq_length, seq_length)
            attention_mask = attention_mask.transpose(0, 1).transpose(1, 2)  # (seq_length, batch_size, seq_length)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)
        
        # Pass through transformer encoder
        embeddings = embeddings.transpose(0, 1)  # Transformer expects (seq_length, batch_size, embed_dim)
        output = self.transformer_encoder(embeddings, src_key_padding_mask=None)  # (seq_length, batch_size, embed_dim)
        output = output.transpose(0, 1)  # Back to (batch_size, seq_length, embed_dim)
        
        return output