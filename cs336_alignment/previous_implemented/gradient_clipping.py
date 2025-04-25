import torch

def gradient_clipping(params, max_norm, eps=1e-6):
    # Compute total L2 norm across all parameter gradients
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.detach().norm(2).item() ** 2

    total_norm = torch.sqrt(torch.tensor(total_norm_sq))

    # Scale gradients if norm exceeds max_norm
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale_factor)
