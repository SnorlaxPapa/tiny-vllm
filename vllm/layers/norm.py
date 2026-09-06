import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    @torch.compile
    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = x.dtype
        # compute rms
        x = x.float() + residual.float()
        residual = x
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        x = x / rms * self.weight
        # add residual
        return x.to(original_dtype), residual.to(original_dtype)

    
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        # compute rms
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        return (x / rms * self.weight).to(original_dtype)