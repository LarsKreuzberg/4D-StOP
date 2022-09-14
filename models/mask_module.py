import torch.nn as nn

from pointnet2 import pytorch_utils as pt_utils
from typing import List

class MaskModule(nn.Module):
    def __init__(self, mlp: List[int], bn: bool = True, use_xyz: bool = True):
        super().__init__()

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, grouped_features):
        proposal_binary_mask = self.mlp_module(grouped_features)
        return proposal_binary_mask