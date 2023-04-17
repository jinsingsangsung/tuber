import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class CSCE(nn.Module):

    def __init__(self, hidden_dim, dim_dynamic, num_dynamic):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params + hidden_dim)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        # pooler_resolution = cfg.CONFIG.MODEL.SparseRCNN.ROI_BOX_HEAD.POOLER_RESOLUTION
        # num_output = self.hidden_dim * pooler_resolution ** 2
        # self.out_layer = nn.Linear(num_output, self.hidden_dim)
        # self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, query_outputs):
        '''
        pro_features: (num_classes, self.d_model)
        query_outputs: lay_n, bs, nb, d
        '''
        lay_n, bs, nb, d = query_outputs.shape
        features = query_outputs.flatten(0,1).permute(1, 0, 2) # nb, lay_n*bs, d
        pro_features = pro_features.unsqueeze(0)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:-self.hidden_dim].view(-1, self.dim_dynamic, self.hidden_dim)
        param3 = parameters[:, :, -self.hidden_dim:].view(-1, self.hidden_dim).unsqueeze(-1)
        features = torch.einsum("nbd,cde->nbce", features, param1)
        # features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.einsum("nbce,ced->nbcd",features, param2)
        features = self.norm2(features)
        features = self.activation(features)
        features = torch.einsum("nbcd,cde->nbce", features, param3).squeeze(-1)
        features = features.permute(1,0,2).view(lay_n, bs, nb, -1)
        # features = self.out_layer(features)
        # features = self.norm3(features)
        # features = self.activation(features)

        return features