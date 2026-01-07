import torch
from thop import profile, clever_format

# Function to compute FLOPS
def compute_flops(model, config):
    prior_image_input = torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(config.device)
    prior_indication_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(config.device)  # Dummy text sequences
    prior_report_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(config.device)  # Dummy text sequences
    current_image_input = torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(config.device)
    current_indication_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(config.device)  # Dummy text sequences
    current_report_output = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(config.device)  # Dummy text sequences
    flops, _ = profile(model, inputs=((prior_image_input, prior_indication_input, prior_report_input, current_image_input, current_indication_input, current_report_output)))
    return flops