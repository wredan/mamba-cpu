# Copyright (c) 2023, Tri Dao.
# Implement residual + rms_norm.

import math
import torch
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))

    def forward(self, x):
        rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rstd * self.weight
