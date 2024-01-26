# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import sys
import numpy as np

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from models.transformer.util.misc import NestedTensor, is_main_process
from models.transformer.position_encoding import build_position_encoding

from models.backbones.ir_CSN_50 import build_CSN
from models.backbones.ir_CSN_152 import build_CSN as build_CSN_152
from models.backbones.vit import build_ViT
from models.backbones.slowfast import build_SlowFast
from models.backbones.slowfast_utils import pack_pathway_output
import fvcore.nn.weight_init as weight_init
# from models.transformer.transformer_layers import LSTRTransformerDecoder, LSTRTransformerDecoderLayer, layer_norm
# from detectron2.layers import ShapeSpec

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, t, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Backbone(nn.Module):

    def __init__(self, train_backbone: bool, num_channels: int, position_embedding, return_interm_layers, cfg):
        super().__init__()

        if cfg.CONFIG.MODEL.BACKBONE_NAME== 'CSN-152':
            print("CSN-152 backbone")
            self.body = build_CSN_152(cfg)
        elif cfg.CONFIG.MODEL.BACKBONE_NAME== 'CSN-50':
            print("CSN-50 backbone")
            self.body = build_CSN(cfg)
        elif cfg.CONFIG.MODEL.BACKBONE_NAME== 'SlowFast':
            print("SlowFast-R101 backbone")
            self.body = build_SlowFast(cfg)
        else:
            print("ViT-B backbone")
            self.body = build_ViT(cfg)
        self.position_embedding = position_embedding

        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        self.ds = cfg.CONFIG.MODEL.SINGLE_FRAME
        # if cfg.CONFIG.MODEL.SINGLE_FRAME:
        #     if cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'avg':
        #         self.pool = nn.AvgPool3d((cfg.CONFIG.DATA.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE, 1, 1))
        #         # print("avg pool: {}".format(cfg.CONFIG.DATA.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE))
        #     elif cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'max':
        #         self.pool = nn.MaxPool3d((cfg.CONFIG.DATA.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE, 1, 1))
        #         print("max pool: {}".format(cfg.CONFIG.DATA.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE))
        #     elif cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'decode':
        #         self.query_pool = nn.Embedding(1, 2048)
        #         self.pool_decoder = LSTRTransformerDecoder(
        #             LSTRTransformerDecoderLayer(d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1), 1,
        #             norm=layer_norm(d_model=2048, condition=True))
        use_ViT = "ViT" in cfg.CONFIG.MODEL.BACKBONE_NAME
        use_SlowFast = "SlowFast" in cfg.CONFIG.MODEL.BACKBONE_NAME
        use_CSN = "CSN" in cfg.CONFIG.MODEL.BACKBONE_NAME
        self.use_ViT = use_ViT
        self.use_SlowFast = use_SlowFast
        self.use_CSN = use_CSN
        if return_interm_layers:
            if use_ViT:
                out_channel = cfg.CONFIG.MODEL.D_MODEL
                in_channels = [cfg.CONFIG.ViT.EMBED_DIM]*4
                self.strides = [8, 16, 32]
                self.num_channels = in_channels
                self.lateral_convs = nn.ModuleList()

                for idx, scale in enumerate([4, 2, 1, 0.5]):
                    dim = in_channels[idx]
                    if scale == 4.0:
                        layers = [
                            nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                            LayerNorm(dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        ]
                        out_dim = dim // 4
                    elif scale == 2.0:
                        layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim // 2
                    elif scale == 1.0:
                        layers = []
                        out_dim = dim
                    elif scale == 0.5:
                        layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim
                    else:
                        raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                    layers.extend(
                        [
                            nn.Conv3d(
                                out_dim,
                                out_channel,
                                kernel_size=1,
                                bias=False,
                            ),
                            LayerNorm(out_channel),
                            nn.Conv3d(
                                out_channel,
                                out_channel,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ]
                    )
                    layers = nn.Sequential(*layers)

                    self.lateral_convs.append(layers)                      
            elif use_SlowFast:
                self.num_pathways = 2       
                out_channel = cfg.CONFIG.MODEL.D_MODEL      
                in_channels = [256+32, 512+64, 1024+128, 2048+256]
                self.num_channels = in_channels
                self.strides = [8, 16, 32]
                self.lateral_convs = nn.ModuleList()
                for idx, in_channel in enumerate(in_channels):
                    lateral_conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)
                    weight_init.c2_xavier_fill(lateral_conv)
                    self.lateral_convs.append(lateral_conv)
                    self.in_features = None
                    
            else:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
                self.in_features = None             
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        if use_CSN:
            self.body = IntermediateLayerGetter(self.body, return_layers=return_layers)
        self.backbone_name = cfg.CONFIG.MODEL.BACKBONE_NAME
        self.temporal_ds_strategy = cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY
        self.cfg = cfg

    def space_forward(self, features):
        mapped_features = {}
        for i, feature in enumerate(features):
            if isinstance(feature, NestedTensor):
                mask = feature.mask
                feature = feature.tensors
                mapped_features.update({f"{i}": NestedTensor(self.lateral_convs[i](feature), mask)})
            else:
                mapped_features.update({f"{i}": self.lateral_convs[i](feature)})
            
        return mapped_features

    def forward(self, tensor_list: NestedTensor):
        if "SlowFast" in self.backbone_name:
            # xs, xt = self.body([tensor_list.tensors[:, :, ::4, ...], tensor_list.tensors])
            x = pack_pathway_output(self.cfg, tensor_list.tensors, pathways=self.num_pathways)
            xs = self.body(x) #interm layer features
            # xs_orig = xt
        elif "TPN" in self.backbone_name:
            xs, xt = self.body(tensor_list.tensors)
            xs_orig = xt
        else:
            xs = self.body(tensor_list.tensors) #interm layer features
            # xs_orig = xs
        # print(xs['0'].shape)
        # print(xs['1'].shape)
        # print(xs['2'].shape)
        # print(xs['3'].shape)
        # bs, ch, t, w, h = xs.shape
        if self.use_ViT or self.use_SlowFast:
            xs = self.space_forward(xs)

        out: Dict[str, NestedTensor] = {}
        # m = tensor_list.mask
        # assert m is not None

        # mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        # mask = mask.unsqueeze(1).repeat(1,xs.shape[2],1,1)

        # out = [NestedTensor(xs, mask)]
        # pos = [self.position_embedding(NestedTensor(xs, mask))]

        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            mask = mask.unsqueeze(1).repeat(1,x.shape[2],1,1)
            # print("mask shape: ", mask.shape)
            # bs, c, t, h, w = x.shape
            # x = x.permute(2,0,1,3,4).reshape(t*bs, c, h, w)
            # mask = mask.permute(1,0,2,3).reshape(t*bs, h, w)
            out[name] = NestedTensor(x, mask)
        return out #, pos #, xs_orig


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        # self.output_shape = {
        #     name: (ShapeSpec(
        #         channels=self.num_channels[l], stride=backbone.strides[l]
        #     ) if (l < len(self.num_channels))
        #         else ShapeSpec(
        #             channels=self.num_channels[-1], stride=backbone.strides[-1]
        #         ))
        #     for l, name in enumerate(backbone.in_features)
        # }

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos #, xl


def build_3d_backbone(cfg):
    position_embedding = build_position_encoding(cfg.CONFIG.MODEL.D_MODEL)
    backbone = Backbone(train_backbone=cfg.CONFIG.TRAIN.LR_BACKBONE > 0, 
                     num_channels=cfg.CONFIG.MODEL.DIM_FEEDFORWARD, 
                     position_embedding=position_embedding, 
                     return_interm_layers=True,
                     cfg=cfg)
    model = Joiner(backbone, position_embedding)
    return model