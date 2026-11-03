import torch
import torch.nn as nn


class DirectMultiStepDecoder(nn.Module):
    def __init__(self, d_model: int, c_out: int, out_len: int):
        super().__init__()
        self.out_len = out_len
        self.out_heads = nn.ModuleList([
            nn.Linear(d_model, c_out) for _ in range(out_len)
        ])

    def forward(self, h: torch.Tensor, last_input: torch.Tensor) -> torch.Tensor:
        deltas = torch.stack([head(h) for head in self.out_heads], dim=1)
        return last_input.unsqueeze(1) + deltas.reshape(
            last_input.size(0), self.out_len, *last_input.shape[1:]
        )
