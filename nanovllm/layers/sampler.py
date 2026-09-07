import torch
import torch.nn as nn


class Sampler(nn.Module):
    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """uses log max to avoid softmax calculation"""

        logits = logits.float()
        greedy_mask = temperatures == 0

        safe_temperatures = torch.where(
            greedy_mask,
            torch.ones_like(temperatures),
            temperatures,
        )

        scaled_logits = logits / safe_temperatures.unsqueeze(1)

        noise = torch.empty_like(logits).exponential_(1)
        noise.clamp_min_(1e-10)

        sampled_scores = scaled_logits - noise.log()

        scores = torch.where(
            greedy_mask.unsqueeze(1),
            logits,
            sampled_scores,
        )

        return scores.argmax(dim=-1)