# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, MSDeformAttn3D


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                            use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
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
            grid = torch.cat([grid_t.unsqueeze(-1), grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_T.unsqueeze(-1), valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 3)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1, -1) + 0.5) / scale
            twh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, twh), -1).view(N_, -1, 6)
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
        valid_ratio = torch.stack([valid_ratio_t, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        Input:
            - srcs: List([bs, c, t, h, w])
            - masks: List([bs, t, h, w])
            - pos_embeds: List([bs, c, t, h, w])
            - query_embed: take either torch.Tensor([nq, d+4]) or torch.Tensor([b, nq, d+4])
        """
        assert self.two_stage or query_embed is not None

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
        # import ipdb; ipdb.set_trace()

        # prepare input for decoder
        bs, _, c = memory.shape

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatio_temporal_shapes)
            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            if query_embed.size(0) != bs:
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid() 
            # bs, num_queries, 2
            init_reference_out = reference_points

        # decoder
        # import ipdb; ipdb.set_trace()
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatio_temporal_shapes, level_start_index, valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


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


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatio_temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (T_, H_, W_) in enumerate(spatio_temporal_shapes):

            ref_t, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
                                          )
            ref_t = ref_t.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * T_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)
            ref = torch.stack((ref_t, ref_x, ref_y), -1)
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
            output = layer(output, pos, reference_points, spatio_temporal_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn3D(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatio_temporal_shapes, level_start_index, src_padding_mask=None):
        # self attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatio_temporal_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 3)
            if self.no_sine_embed:
                self.ref_point_head = MLP(6, d_model, d_model, 4)
            else:
                self.ref_point_head = MLP(3 * d_model, d_model, d_model, 3)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 3)


    def forward(self, tgt, reference_points, src, src_spatio_temporal_shapes,       
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        bs = src.shape[0]
        if reference_points.size(0) != bs:
            reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                z = t = torch.full((reference_points.size(0), reference_points.size(1), 1), 0.5, device=reference_points.device)
                reference_points = torch.cat([z, reference_points[:,:,:2], t, reference_points[:,:,2:]], -1)
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 6
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)                 


            output = layer(output, query_pos, reference_points_input, src, src_spatio_temporal_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(cfg):
    return DeformableTransformer(
        d_model=cfg.CONFIG.MODEL.D_MODEL,
        nhead=cfg.CONFIG.MODEL.NHEAD,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=cfg.CONFIG.MODEL.DIM_FEEDFORWARD,
        dropout=cfg.CONFIG.MODEL.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=cfg.CONFIG.MODEL.NUM_FEATURE_LEVELS,
        dec_n_points=cfg.CONFIG.MODEL.NUM_DEC_N_POINTS,
        enc_n_points=cfg.CONFIG.MODEL.NUM_ENC_N_POINTS,
        two_stage=False,
        two_stage_num_proposals=cfg.CONFIG.MODEL.QUERY_NUM,
        use_dab=True)


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
    z_embed = pos_tensor[:, :, 0] * scale
    x_embed = pos_tensor[:, :, 1] * scale
    y_embed = pos_tensor[:, :, 2] * scale
    pos_z = z_embed[:, :, None] / dim_t
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 3:
        pos = torch.cat((pos_t, pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 6:
        t_embed = pos_tensor[:, :, 3] * scale
        pos_t = t_embed[:, :, None] / dim_t
        pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)

        w_embed = pos_tensor[:, :, 3] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_z, pos_y, pos_x, pos_t, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos