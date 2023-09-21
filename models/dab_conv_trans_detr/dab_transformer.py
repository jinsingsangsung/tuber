# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List
from utils.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch import nn, Tensor
from .attention import MultiheadAttention

from models.transformer.util.misc import NestedTensor
from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, MSDeformAttn3D
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as checkpoint

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3,3), padding=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.conv2 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.conv3 = nn.Linear(4*dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
    
    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.permute(0,3,1,2)
        x = input + self.drop_path(x)
        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_feature_levels=4,
                 enc_n_points=8,
                 two_stage=False,
                 two_stage_num_proposals=300,
                 high_dim_query_update=False,
                 no_sine_embed=False,
                 gradient_checkpointing=False,
                 num_conv_blocks=3,
                 ):

        super().__init__()

        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, gradient_checkpointing)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        cls_decoder_layer = TransformerClassDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, num_conv_blocks)        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                          gradient_checkpointing=gradient_checkpointing,
                                          )
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.num_feature_levels = num_feature_levels
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)
            
        self.gradient_checkpointing = gradient_checkpointing
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn3D):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatio_temporal_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (T_, H_, W_) in enumerate(spatio_temporal_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + T_ * H_ * W_)].view(N_, T_, H_, W_, 1)
            valid_T = torch.sum(~mask_flatten_[:, :, 0, 0, 0], 1)
            valid_H = torch.sum(~mask_flatten_[:, 0, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, 0, :, 0], 1)

            grid_t, grid_y, grid_x = torch.meshgrid(torch.linspace(0, T_ - 1, T_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], grid_t.unsqueeze(-1), -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], valid_T.unsqueeze(-1), 1).view(N_, 1, 1, 3)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1, -1) + 0.5) / scale
            wht = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wht), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += (T_ * H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask): # using mask, see the valid t,h,w area in flattened space
        _, T, H, W = mask.shape
        valid_T = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)
        valid_ratio_t = valid_T.float() / T
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_t], -1)
        return valid_ratio

    def make_interpolated_features(self, features, pos=None, num_frames=32, level=0):
        '''
            features: list of tensors [B, C, T, H_l, W_l]
                    l = 0, 1, ..., num_feats-1
            return: list of tensors [B, C, T, H_0, W_0]
        '''
        interpolated_features = []
        n_levels = len(features)
        assert level < n_levels-1, f"target feature level {level} should be less than the number of feature level {len(features)}"
        use_mask = isinstance(features[0], NestedTensor)
        if use_mask:
            tensors = [feature.tensors for feature in features]
            mask = features[level].mask
            mask = mask.repeat(1, num_frames//mask.size(1), 1, 1)
            features = tensors

        B, C, T, H, W = features[level].shape
        if T == num_frames:
            dh = torch.linspace(-1, 1, H, device=features[0].device)
            dw = torch.linspace(-1, 1, W, device=features[0].device)
            meshy, meshx = torch.meshgrid((dh, dw))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid[None].expand(B*T, *grid.size())
            for l, feature in enumerate(features):
                feature = rearrange(feature, 'b c t h w -> (b t) c h w')
                interpolated_features.append(
                    F.grid_sample(
                        feature, grid,
                        mode='bilinear', padding_mode='zeros', align_corners=False,            
                    ).view(B,T,C,H,W).transpose(1,2).contiguous()
                )
        else:
            dh = torch.linspace(-1, 1, H, device=features[0].device)
            dw = torch.linspace(-1, 1, W, device=features[0].device)
            dt = torch.linspace(-1, 1, num_frames, device=features[0].device)
            mesht, meshy, meshx= torch.meshgrid((dt, dh, dw))
            grid = torch.stack((meshx, meshy, mesht), -1)
            grid = grid[None].expand(B, *grid.size())
            for l, feature in enumerate(features):
                interpolated_features.append(
                    F.grid_sample(
                        feature, grid,
                        mode='bilinear', padding_mode='zeros', align_corners=False,
                    )
                )
        if use_mask:
            if not pos is None:
                pos[level] = pos[level].repeat(1, 1, num_frames//pos[level].size(2), 1, 1)
                return [NestedTensor(inter_feat, mask) for inter_feat in interpolated_features], [pos[level]]*n_levels
            else:
                return [NestedTensor(inter_feat, mask) for inter_feat in interpolated_features], None
        if not pos is None:
            pos[level] = pos[level].repeat(1, 1, num_frames//pos[level].size(2), 1, 1)
            return interpolated_features, [pos[level]]*n_levels
        else:
            return interpolated_features, None

    def forward(self, srcs, masks, pos_embeds, refpoint_embed=None):
        """
        Input:
            - srcs: List([bs, c, t, h, w])
            - masks: List([bs, t, h, w])
            - pos_embeds: List([bs, c, t, h, w])
            - refpoint_embed: take either torch.Tensor([nq, d+4]) or torch.Tensor([b, nq, d+4])
        """ 
        assert self.two_stage or refpoint_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatio_temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t, h, w = src.shape
            spatio_temporal_shape = (t, h, w)
            spatio_temporal_shapes.append(spatio_temporal_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, thw, c
            mask = mask.flatten(1)                              # bs, thw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, thw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{txhxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{txhxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatio_temporal_shapes = torch.as_tensor(spatio_temporal_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatio_temporal_shapes.new_zeros((1, )), spatio_temporal_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatio_temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        
        # revert to the original shape
        srcs_per_lvl = []
        poses_per_lvl = []
        for i, (index, shape) in enumerate(zip(level_start_index, spatio_temporal_shapes)):
            if i < self.num_feature_levels-1:
                src_l = memory[:, index:level_start_index[i+1], :]
                pos_l = lvl_pos_embed_flatten[:, index:level_start_index[i+1], :]
            else:
                src_l = memory[:, index:, :]
                pos_l = lvl_pos_embed_flatten[:, index:, :]
            src_l = src_l.reshape(-1, *shape, self.d_model).permute(0,4,1,2,3)
            pos_l = pos_l.reshape(-1, *shape, self.d_model).permute(0,4,1,2,3)
            srcs_per_lvl.append(NestedTensor(src_l, masks[i]))
            poses_per_lvl.append(pos_l)
        
        features_per_lvl, poses_per_lvl = self.make_interpolated_features(srcs_per_lvl, poses_per_lvl, level=-2)
        # bs, c, t, h, w = src.shape
        # src_shape = src.shape
        # src = src.permute(0,2,1,3,4).contiguous()
        # src = src.reshape(bs, t, c, h, w).flatten(0,1) # bs*t, c, h, w
        # src = src.flatten(2).permute(2, 0, 1) # hw, bst, c
        # pos_embed = pos_embed.permute(0,2,1,3,4).contiguous().flatten(0,1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.repeat(1, bs, 1) #n_q, bs * t, 4
        # mask = mask.flatten(0,1).flatten(1)

        # memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, src_shape=src_shape) 
        # temporal dimension is alive
        # query_embed = gen_sineembed_for_position(refpoint_embed)
        intpltd_srcs_per_lvl = []
        intpltd_masks_per_lvl = []
        for feature in features_per_lvl:
            intpltd_srcs_per_lvl.append(feature.tensors)
            intpltd_masks_per_lvl.append(feature.mask)
        
        srcs_per_lvl = torch.stack(intpltd_srcs_per_lvl, dim=-1) # bs, c, t, h, w, l
        masks_per_lvl = torch.stack(intpltd_masks_per_lvl, dim=-1)
        poses_per_lvl = torch.stack(poses_per_lvl, dim=-1) # bs, c, t, h, w, l

        bs, c, t, h, w, l = srcs_per_lvl.shape
        num_queries = refpoint_embed.shape[0]
        
        if self.eff:
            memory = srcs_per_lvl[:,:,t//2:t//2+1,:,:,:]
            pos_embed = poses_per_lvl[:,:,t//2:t//2+1,:,:,:]
            mask = masks_per_lvl[:,t//2:t//2+1,:,:,:]
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            memory = srcs_per_lvl
            pos_embed = poses_per_lvl
            mask = masks_per_lvl
            tgt = torch.zeros(num_queries, bs*t, self.d_model, device=refpoint_embed.device)
        
        # prepare input for the decoder
        memory = rearrange(memory, "B C T H W L -> L (H W) (B T) C")
        pos_embed = rearrange(pos_embed, "B C T H W L -> L (H W) (B T) C")
        mask = rearrange(mask, "B T H W L -> (B T) (H W) L")[..., 0]

        hs, cls_hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed, orig_res=(h,w))
        return hs, cls_hs, references


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, gradient_checkpointing=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.gradient_checkpointing=gradient_checkpointing

    @staticmethod
    def get_reference_points(spatio_temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (T_, H_, W_) in enumerate(spatio_temporal_shapes):

            ref_t, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
                                          )
            ref_t = ref_t.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * T_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y, ref_t), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatio_temporal_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(ti*hi*wi), 256]
            - spatio_temporal_shapes: h,w of each level [num_level, 3]
            - level_start_index: [num_level] start point of level in sum(ti*hi*wi).
            - valid_ratios: [bs, num_level, 3]
            - pos: pos embed for src. [bs, sum(ti*hi*wi), 256]
            - padding_mask: [bs, sum(ti*hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(ti*hi*wi), num_level, 2]
        """
        output = src
        # bs, sum(ti*hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatio_temporal_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            if self.gradient_checkpointing:
                def custom_layer(module):
                    def custom_forward(*inputs):
                        return module(inputs[0],
                                      pos = inputs[1],
                                      reference_points = inputs[2],
                                      spatio_temporal_shapes = inputs[3],
                                      level_start_index = inputs[4],
                                      padding_mask = inputs[5])
                    return custom_forward
                output = checkpoint.checkpoint(custom_layer(layer), output, pos, reference_points, spatio_temporal_shapes, level_start_index, padding_mask)
            else:
                output = layer(output, pos, reference_points, spatio_temporal_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn3D(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatio_temporal_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatio_temporal_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_shape = None):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales, src_shape=src_shape)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, cls_decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    gradient_checkpointing=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.cls_layers = _get_clones(cls_decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

        self.cls_norm = nn.LayerNorm(d_model)
        self.class_queries = nn.Embedding(80, 256).weight
        
        self.cls_norm2 = nn.LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                orig_res = None,
                ):
        output = tgt

        intermediate = []
        cls_intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()        

        for layer_id, (layer, cls_layer) in enumerate(zip(self.layers, self.cls_layers)):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed) 

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs*t, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            if self.gradient_checkpointing:
                def custom_layer(module):
                    def custom_forward(*inputs):
                        return module(inputs[0],
                                      inputs[1],
                                      tgt_mask = inputs[2],
                                      memory_mask = inputs[3],
                                      tgt_key_padding_mask = inputs[4],
                                      memory_key_padding_mask = inputs[5],
                                      pos = inputs[6],
                                      query_pos = inputs[7],
                                      query_sine_embed = inputs[8],
                                      is_first = inputs[9])
                    return custom_forward
                def custom_layer2(module):
                    def custom_forward2(*inputs):
                        return module(*inputs)
                    return custom_forward2
                output, actor_feature = checkpoint.checkpoint(custom_layer(layer),
                                               output,
                                               memory,
                                               tgt_mask,
                                               memory_mask,
                                               tgt_key_padding_mask,
                                               memory_key_padding_mask,
                                               pos,
                                               query_pos,
                                               query_sine_embed,
                                               (layer_id==0))
                cls_output = checkpoint.checkpoint(custom_layer2(cls_layer),
                                                   actor_feature.clone().detach(),
                                                   memory,
                                                   pos,
                                                   query_sine_embed,
                                                   self.class_queries,
                                                   orig_res,
                                                   len(tgt))
            else:
                output, actor_feature = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                            is_first=(layer_id == 0))

                cls_output = cls_layer(actor_feature.clone().detach(), memory, pos, query_sine_embed, self.class_queries, orig_res, len(tgt))
            
            if layer_id != 0:
                cls_output = self.cls_norm(cls_output + prev_output)
            prev_output = cls_output
            
            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                cls_intermediate.append(self.cls_norm2(cls_output))

        if self.norm is not None:
            output = self.norm(output)
            cls_output = self.cls_norm2(cls_output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                cls_intermediate.pop()
                cls_intermediate.append(cls_output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(cls_intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    torch.stack(cls_intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module): # spatially, temporally

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1_s = nn.Linear(d_model, dim_feedforward)
        self.dropout_s = nn.Dropout(dropout)
        self.linear2_s = nn.Linear(dim_feedforward, d_model)

        self.norm1_s = nn.LayerNorm(d_model)
        self.norm2_s = nn.LayerNorm(d_model)
        self.dropout1_s = nn.Dropout(dropout)
        self.dropout2_s = nn.Dropout(dropout)

        self.linear1_t = nn.Linear(d_model, dim_feedforward)
        self.dropout_t = nn.Dropout(dropout)
        self.linear2_t = nn.Linear(dim_feedforward, d_model)

        self.norm1_t = nn.LayerNorm(d_model)
        self.norm2_t = nn.LayerNorm(d_model)
        self.dropout1_t = nn.Dropout(dropout)
        self.dropout2_t = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     src_shape = None):
        b, c, t, h, w = src_shape # original src shape
        # current src shape: hw, bt, c
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn_s(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1_s(src2)
        src = self.norm1_s(src)
        src2 = self.linear2_s(self.dropout_s(self.activation(self.linear1_s(src))))
        src = src + self.dropout2_s(src2)
        src = self.norm2_s(src) # spatially encoded
        
        # temporally encode
        src = src.reshape(-1,b,t,c).permute(2,1,0,3).contiguous().flatten(1,2) # t, b*hw, c
        pos = pos.reshape(-1,b,t,c).permute(2,1,0,3).contiguous().flatten(1,2)
        q = k = self.with_pos_embed(src, pos)
        # src_key_padding_mask_t = torch.zeros((b*h*w, t), device = src_key_padding_mask.device)>0
        src2 = self.self_attn_t(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=None)[0]
        src = src + self.dropout1_t(src2)
        src = self.norm1_t(src)
        src2 = self.linear2_t(self.dropout_t(self.activation(self.linear1_t(src))))
        src = src + self.dropout2_t(src2)
        src = self.norm2_t(src) # spatially encoded

        # make it to the original input shape so that it can iterate
        src = src.reshape(t, b, h*w, c).permute(2,1,0,3).contiguous().flatten(1,2)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, num_levels=4,):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.lvl_w_embed = nn.Linear(d_model, num_levels)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model, query_specific_key=True)

        self.query_specific_key = True
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        lvl_w = self.lvl_w_embed(tgt) # num_queries, BT, num_levels
        q_memory = torch.einsum("ntl,lhtc->nhtc", lvl_w, memory) # (N_q, BT, L), (L, HW, BT, C),  ->  N_q, HW, BT, C
        if self.query_specific_key:
            memory = q_memory
        else:
            memory = memory.mean(0)
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw = k_content.shape[-3]
        if self.query_specific_key:
            k_pos = self.ca_kpos_proj(pos)[0:1].expand(num_queries, -1, -1, -1)
        else:
            k_pos = self.ca_kpos_proj(pos)[0]

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        if self.query_specific_key:
            k = k.view(num_queries, hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(num_queries, hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=4).view(num_queries, hw, bs, n_model * 2)
        else:
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt_temp = tgt
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, tgt_temp, q_memory

class TransformerClassDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", num_conv_blocks=3):
        super().__init__()
        
        self.d_model = d_model
        
        # Separate FFN layer
        self.cls_linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout) # self.dropout
        self.cls_linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout) # self.dropout
        self.cls_norm = nn.LayerNorm(d_model)

        # Actor-centric convolution layers
        self.conv_norm = nn.LayerNorm(d_model)
        conv_block = ConvBlock(d_model, 0)
        self.conv_blocks = nn.ModuleList([conv_block for _ in range(num_conv_blocks)])   
        
        # Query self-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.dropout3 = nn.Dropout(dropout) # self.dropout1
        self.norm1 = nn.LayerNorm(d_model)
        
        # Class decoder cross-attention
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.cls_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        
        # FFN layer
        # self.linear = nn.Linear(d_model, d_model)
        self.cls_linear1_ = nn.Linear(d_model, dim_feedforward)
        self.dropout1_ = nn.Dropout(dropout) # self.dropout_
        self.cls_linear2_ = nn.Linear(dim_feedforward, d_model)
        self.dropout2_ = nn.Dropout(dropout) # self.dropout_
        self.cls_norm_ = nn.LayerNorm(d_model)

    def forward(self, actor_feature, memory, pos, query_sine_embed, class_queries, orig_res, num_queries):
                    
        # separate classification branch from localization
        actor_feature2 = self.cls_linear2(self.dropout1(self.activation(self.cls_linear1(actor_feature))))
        actor_feature = actor_feature + self.dropout2(actor_feature2)
        actor_feature = self.cls_norm(actor_feature)

        # apply actor-centric convolution
        h, w = orig_res
        actor_feature_expanded = actor_feature.flatten(0,1)[..., None, None].expand(-1, -1, h, w) # N_q*B, D, H, W
        encoded_feature_expanded = memory[:, None].expand(-1, num_queries, -1, -1).flatten(1,2).view(h,w,-1,actor_feature.shape[-1]).permute(2,3,0,1) # N_q*B, D, H, W
        cls_feature = actor_feature_expanded + encoded_feature_expanded
        cls_feature = self.conv_norm(cls_feature.transpose(-1, 1).contiguous()).transpose(-1, 1).contiguous()            
        for block in self.conv_blocks:    
        #     if self.gradient_checkpointing:
        #         def custom_conv(module):
        #             def custom_conv_fwd(inputs):
        #                 return module(inputs)
        #             return custom_conv_fwd
        #         cls_feature = checkpoint.checkpoint(custom_conv(block), cls_feature)
        #     else:
        #         cls_feature = block(cls_feature)
            cls_feature = block(cls_feature)
        
        # class query self-attention
        query = class_queries[:, None].expand(-1, actor_feature_expanded.shape[0], -1)
        query2 = self.self_attn(query, query, query)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        # class decoder cross-attention
        key = torch.cat([self.k_proj(cls_feature).flatten(2).permute(2,0,1), pos[:,None].expand(-1, num_queries, -1, -1).flatten(1,2)], dim=-1)
        cls_query_pos = self.cls_qpos_sine_proj(query_sine_embed).flatten(0,1)[None].expand(len(class_queries), -1 ,-1)
        query = torch.cat([query, cls_query_pos], dim=-1)
        value = self.v_proj(encoded_feature_expanded).flatten(2).permute(2,0,1)
        # if self.gradient_checkpointing:
        #     def custom_layer(module):
        #         def custom_forward(*inputs):
        #             return module(query = inputs[0],
        #                             key = inputs[1],
        #                             value = inputs[2]
        #                             )
        #         return custom_forward
        #     cls_output = checkpoint.checkpoint(custom_layer(self.cross_attn), query, key, value)[0].reshape(len(class_queries), num_queries, -1, self.d_model).permute(1,2,0,3)
        # else:
        #     cls_output = self.cross_attn(query=query, key=key, value=value)[0].reshape(len(class_queries), num_queries, -1, self.d_model).permute(1,2,0,3)
        cls_output = self.cross_attn(query=query, key=key, value=value)[0].reshape(len(class_queries), num_queries, -1, self.d_model).permute(1,2,0,3)
        
        # FFN
        cls_output2 = self.cls_linear2_(self.dropout1_(self.activation(self.cls_linear1_(cls_output))))
        cls_output = cls_output + self.dropout2_(cls_output2)
        cls_output = self.cls_norm_(cls_output)
        
        return cls_output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.CONFIG.MODEL.D_MODEL,
        dropout=cfg.CONFIG.MODEL.DROPOUT,
        nhead=cfg.CONFIG.MODEL.NHEAD,
        num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
        dim_feedforward=cfg.CONFIG.MODEL.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.CONFIG.MODEL.ENC_LAYERS,
        num_decoder_layers=cfg.CONFIG.MODEL.DEC_LAYERS,
        normalize_before=cfg.CONFIG.MODEL.NORMALIZE_BEFORE,
        return_intermediate_dec=True,
        query_dim=4,
        activation="relu",
        num_patterns=cfg.CONFIG.MODEL.NUM_PATTERNS,
        bbox_embed_diff_each_layer=cfg.CONFIG.MODEL.BBOX_EMBED_DIFF_EACH_LAYER,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")