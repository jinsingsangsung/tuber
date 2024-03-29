# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
The code refers to https://github.com/facebookresearch/detr
Modified by Zhang Yanyi
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, position_embedding='sine',
                 num_feat = 9):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.position_embedding = position_embedding
        self.num_feat = num_feat
        print("using refined transformer")


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, t, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory, attn_encoder = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, attn_decoder = self.decoder(self.num_feat, tgt, memory, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, t, h, w), attn_encoder, attn_decoder


class Transformer2(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, position_embedding='sine',
                 num_feat = 9):
        super().__init__()

        decoder_layer = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.position_embedding = position_embedding
        self.num_feat = num_feat
        print("using refined transformer")


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, attn_weight, mask_p=0.7):
        # flatten NxCxHxW to HWxNxC
        nb, d = query_embed.shape
        bs, c, t, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1).contiguous() # thw, bs, c
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).contiguous() # thw, bs, c
        mask = mask.flatten(1).permute(1, 0).contiguous().unsqueeze(-1) # thw, bs
        
        thw = src.size(0)
        new_thw = int(mask_p*thw)
        
        filtered_indices = attn_weight.topk(new_thw, -1, largest=False, sorted=False)[1]
        filtered_indices = filtered_indices.flatten(0,1)
        
        src = src.unsqueeze(0).repeat(nb, 1, 1, 1).permute(2, 0, 1, 3).contiguous().flatten(0,1) # bs*nb, thw, c
        pos_embed = pos_embed.unsqueeze(0).repeat(nb, 1, 1, 1).permute(2, 0, 1, 3).contiguous().flatten(0,1) # bs*nb, thw, c
        mask = mask.unsqueeze(0).repeat(nb, 1, 1, 1).permute(2, 0, 1, 3).contiguous().flatten(0,1) # bs*nb, thw,
        src = torch.stack([src[..., i].gather(1, filtered_indices) for i in range(c)], -1).view(bs, -1, c) # bs, nb*thw, c
        pos_embed = torch.stack([pos_embed[..., i].gather(1, filtered_indices) for i in range(c)], -1).view(bs, -1, c)
        mask = torch.stack([mask[..., 0].gather(1, filtered_indices)], -1).view(bs, -1) #bs, nb*thw
        
        src = src.permute(1,0,2).contiguous()
        pos_embed = pos_embed.permute(1,0,2).contiguous()

        attn_mask = torch.ones(nb, src.size(0), device=query_embed.device) > 0
        for n in range(nb):
            attn_mask[n, n*new_thw:n*new_thw+new_thw] = False

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)

        # memory, attn_encoder = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, attn_decoder = self.decoder(self.num_feat, tgt, src, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed, memory_mask=attn_mask)
        return hs.transpose(1, 2), attn_decoder




class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output, attn = layer(output, src_mask=mask,
                                 src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, num_feat, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output, attn = layer(num_feat, output, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn

        return output, attn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        sa_out = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = sa_out[0]
        attn = sa_out[1]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, num_feat, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        query=self.with_pos_embed(tgt, query_pos)
        bn, bs, dim = query.shape

        key = self.with_pos_embed(memory, pos)
        tgt2, attn = self.multihead_attn(query=query,
                                   key=key,
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn

    def forward_pre(self, num_feat, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, num_feat, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(num_feat, tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(num_feat, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, num_feat, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        query=self.with_pos_embed(tgt, query_pos)
        bn, bs, dim = query.shape

        key = self.with_pos_embed(memory, pos)

        tgt2, attn = self.multihead_attn(query=query,
                                   key=key,
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn

    def forward_pre(self, num_feat, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, num_feat, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(num_feat, tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(num_feat, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


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


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.CONFIG.MODEL.D_MODEL,
        dropout=cfg.CONFIG.MODEL.DROPOUT,
        nhead=cfg.CONFIG.MODEL.NHEAD,
        dim_feedforward=cfg.CONFIG.MODEL.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.CONFIG.MODEL.ENC_LAYERS,
        num_decoder_layers=cfg.CONFIG.MODEL.DEC_LAYERS,
        normalize_before=cfg.CONFIG.MODEL.NORMALIZE_BEFORE,
        return_intermediate_dec=True,
        num_feat = cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE if not cfg.CONFIG.MODEL.SINGLE_FRAME else 1
    )


def build_transformer2(cfg):
    return Transformer2(
        d_model=cfg.CONFIG.MODEL.D_MODEL,
        dropout=cfg.CONFIG.MODEL.DROPOUT,
        nhead=cfg.CONFIG.MODEL.NHEAD,
        dim_feedforward=cfg.CONFIG.MODEL.DIM_FEEDFORWARD,
        # num_encoder_layers=cfg.CONFIG.MODEL.ENC_LAYERS,
        num_decoder_layers=cfg.CONFIG.MODEL.DEC_LAYERS2,
        normalize_before=cfg.CONFIG.MODEL.NORMALIZE_BEFORE,
        return_intermediate_dec=True,
        num_feat = cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE if not cfg.CONFIG.MODEL.SINGLE_FRAME else 1
    )

if __name__ == '__main__':
    from pipelines.video_action_recognition_config import get_cfg_defaults
    import argparse
    parser = argparse.ArgumentParser(description='Train ava transformer models.')
    parser.add_argument('--config-file',
                        default='/workplace/code/AVA_Clean/configuration/Action_DETR_AVA_Local.yaml',
                        help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    transformer = build_transformer(cfg)
    transformer(torch.randn(1, 256, 9, 7, 7))

