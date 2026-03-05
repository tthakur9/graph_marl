import torch
import torch.nn as nn

class IdentityEncoder(nn.Module):
    def forward(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # obs_dict[agent] = (B, obs_dim)
        return obs_dict
