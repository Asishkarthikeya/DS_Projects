import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# Generate Captions Method
def generate_captions(model, prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, temperature=1.0, decoding_strategy='greedy', top_p=None):
    """
    Generates captions for a batch of images using the trained model.

    Args:
        model (nn.Module): The trained image captioning model.
        prior_images (torch.Tensor): A batch of prior image tensors (preprocessed).
        prior_indication_tokens (torch.Tensor): A batch of prior indication token tensors.
        prior_report_tokens (torch.Tensor): A batch of prior report token tensors.
        current_images (torch.Tensor): A batch of current image tensors (preprocessed).
        curr_indication_tokens (torch.Tensor): A batch of current indication token tensors.
        temperature (float): Temperature for scaling logits (default: 1.0).
        decoding_strategy (str): Decoding method ('greedy' or 'sampling', default: 'greedy').
        top_p (float, optional): Top-p threshold for nucleus sampling (default: None).

    Returns:
        list of str: List of generated captions for each image in the batch.
    """
    model.eval()  # Set the model to evaluation mode
    generated_captions = []

    with torch.no_grad():
        # Ensure images and tokens are on the same device as the model
        prior_images = prior_images.to(model.config.device)
        prior_indication_tokens = prior_indication_tokens.to(model.config.device)
        prior_report_tokens = prior_report_tokens.to(model.config.device)
        current_images = current_images.to(model.config.device)
        curr_indication_tokens = curr_indication_tokens.to(model.config.device)

        # Start with the <start> token for each image in the batch
        batch_size = prior_images.size(0)
        caption_tokens = torch.full((batch_size, 1), model.tokenizer.word2idx["<start>"], dtype=torch.long, device=model.config.device)

        for _ in range(model.config.max_seq_len):
            # Forward pass through the model for the entire batch
            outputs, _, _, _, _, _, _, _, _, _ = model(
                prior_images, prior_indication_tokens, prior_report_tokens,
                current_images, curr_indication_tokens, caption_tokens,
                temperature=temperature
            )  # Shape: (batch_size, seq_len, vocab_size)
            next_token_logits = outputs[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Select next token based on decoding strategy
            if decoding_strategy == 'greedy':
                next_token = next_token_logits.argmax(dim=-1)  # Shape: (batch_size,)
                if next_token.dim() == 0:  # Handle batch_size=1
                    next_token = next_token.unsqueeze(0)  # Shape: (1,)
            elif decoding_strategy == 'sampling':
                probs = F.softmax(next_token_logits, dim=-1)  # Shape: (batch_size, vocab_size)
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs <= top_p
                    # Ensure mask is applied per batch
                    next_token = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
                    for i in range(batch_size):
                        sampled_probs_i = sorted_probs[i][mask[i]]
                        sampled_indices_i = sorted_indices[i][mask[i]]
                        if sampled_probs_i.sum() > 0:
                            sampled_probs_i = sampled_probs_i / sampled_probs_i.sum()  # Renormalize
                            sampled_idx = torch.multinomial(sampled_probs_i, num_samples=1)
                            next_token[i] = sampled_indices_i[sampled_idx]
                        else:
                            # Fallback to standard sampling
                            next_token[i] = torch.multinomial(probs[i], num_samples=1)
                else:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: (batch_size,)
                    if next_token.dim() == 0:  # Handle batch_size=1
                        next_token = next_token.unsqueeze(0)  # Shape: (1,)
            else:
                raise ValueError(f"Unsupported decoding_strategy: {decoding_strategy}")

            # Append the predicted token to the captions for each image
            caption_tokens = torch.cat([caption_tokens, next_token.unsqueeze(-1)], dim=-1)

            # Stop if the <end> token is generated for each image in the batch
            if (next_token == model.tokenizer.word2idx["<end>"]).all():
                break

        # Convert token indices to words and join them into sentences
        for i in range(batch_size):
            caption = " ".join(
                model.tokenizer.idx2word[token.item()] 
                for token in caption_tokens[i, 1:] 
                if token.item() != model.tokenizer.word2idx["<end>"]
            )
            generated_captions.append(caption)
        
        return generated_captions

def evaluate_model(model, config, test_loader):
    model.eval()
    references, hypotheses = [], []
    actual_predicted_samples = []

    to_pil = ToPILImage()  # Convert tensor to PIL Image

    with torch.no_grad():
        for prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens in tqdm(test_loader, desc="Evaluating"):
            prior_images = prior_images.to(model.config.device)
            prior_indication_tokens = prior_indication_tokens.to(model.config.device)
            prior_report_tokens = prior_report_tokens.to(model.config.device)
            current_images = current_images.to(model.config.device)
            curr_indication_tokens = curr_indication_tokens.to(model.config.device)
            current_report_tokens = current_report_tokens.to(model.config.device)
    
            # Generate captions for the batch with temperature and decoding strategy
            generated_captions = generate_captions(
                model, prior_images, prior_indication_tokens, prior_report_tokens,
                current_images, curr_indication_tokens,
                temperature=getattr(config, 'temperature', 1.0),
                decoding_strategy=getattr(config, 'decoding_strategy', 'greedy'),
                top_p=getattr(config, 'top_p', None)
            )
    
            # Prepare actual captions
            actual_captions = [
                " ".join(
                    model.tokenizer.idx2word[token.item()] 
                    for token in current_report_tokens[i, 1:] 
                    if token.item() not in [model.tokenizer.word2idx["<pad>"], model.tokenizer.word2idx["<end>"]]
                )
                for i in range(len(current_report_tokens))
            ]
    
            hypotheses.extend(generated_captions)
            references.extend(actual_captions)

            # Collect sample pairs with image (or index)
            for i in range(len(generated_captions)):
                p_kwds = model.tokenizer.decode(prior_indication_tokens[i].tolist())
                p_r = model.tokenizer.decode(prior_report_tokens[i].tolist())
                c_kwds = model.tokenizer.decode(curr_indication_tokens[i].tolist())
                
                actual_predicted_samples.append({
                    "prior_indocation": p_kwds,
                    "prior_report": p_r,
                    "current_indication": c_kwds,
                    "actual_caption": actual_captions[i],
                    "predicted_caption": generated_captions[i]
                })
    
    return references, hypotheses, actual_predicted_samples