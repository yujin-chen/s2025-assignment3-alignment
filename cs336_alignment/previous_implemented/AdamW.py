from typing import Optional, Callable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                    state["v"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                    state["t"] = 0

                m, v, t = state["m"], state["v"], state["t"]
                t += 1  # Increment step count

                # Compute first and second moment estimates
                m = beta1 * m + (1 - beta1) * grad  
                v = beta2 * v + (1 - beta2) * (grad ** 2)  

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                adjusted_lr = lr * (bias_correction2 ** 0.5) / bias_correction1

                # AdamW update
                p.data -= adjusted_lr * m / ((v ** 0.5) + eps)

                #weight decay
                p.data -= lr * weight_decay * p.data  

                # Store updated values
                state["m"], state["v"], state["t"] = m, v, t

        return loss
