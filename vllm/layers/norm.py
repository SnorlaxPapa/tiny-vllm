import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))


    @torch.compile
    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = x.dtype
        # compute rms
        x = x.float() + residual.float()
        residual = x
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        x = x / rms * self.weight
        
        return x.to(original_dtype), residual.to(original_dtype)

    
    @torch.compile
    def non_res_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        # compute rms
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        return (x / rms * self.weight).to(original_dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if residual is not None:
            return self.forward_residual(x, residual)
        else:
            return self.non_res_forward(x)