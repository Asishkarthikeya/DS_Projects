
# Function to compute trainable, non-trainable, and total parameters
def compute_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # Include registered buffers as non-trainable
    buffer_params = sum(b.numel() for b in model.buffers())
    non_trainable_params += buffer_params
    total_params = trainable_params + non_trainable_params
    return trainable_params, non_trainable_params, total_params

