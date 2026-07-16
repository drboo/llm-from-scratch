import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction).

    Formula: y = x * w / sqrt(mean(x²) + eps)

    Pre-norm (applied before attention/FFN) keeps gradients flowing cleanly
    through the residual stream — the residual path is never scaled, so
    gradients reach early layers without vanishing through normalisation ops.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(mean(x²) + eps) == 1 / sqrt(mean(x²) + eps)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
