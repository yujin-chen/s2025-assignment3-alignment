import torch

def cross_entropy(logits, targets):

    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    log_target_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    loss = -log_target_probs.squeeze(-1)
    return loss.mean()


