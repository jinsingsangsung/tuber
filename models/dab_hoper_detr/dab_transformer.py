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
from torch import nn, Tensor
from .attention import MultiheadAttention
from torch.nn.init import constant_

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


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 use_cls_sa=False,
                 cut_gradient=False,
                 more_offset=False,
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        cls_decoder_layer = TransformerClsDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos, rm_self_attn_decoder= not use_cls_sa)
        decoder_norm = nn.LayerNorm(d_model)
        cls_decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, cls_decoder_layer, num_decoder_layers, decoder_norm, cls_decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                          cut_gradient=cut_gradient,
                                          more_offset=more_offset,)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.num_dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)
        self.more_offset = more_offset

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, t, h, w = src.shape
        src_shape = src.shape
        src = src.permute(0,2,1,3,4).contiguous()
        src = src.reshape(bs, t, c, h, w).flatten(0,1) # bs*t, c, h, w
        src = src.flatten(2).permute(2, 0, 1) # hw, bst, c
        pos_embed = pos_embed.permute(0,2,1,3,4).contiguous().flatten(0,1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.repeat(1, bs, 1) #n_q, bs * t, 4
        mask = mask.flatten(0,1).flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, src_shape=src_shape) 
        # temporal dimension is alive
        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]
        if self.eff:
            memory = memory.reshape(-1, bs, t, c)[:,:,t//2:t//2+1,:].flatten(1,2)
            pos_embed = pos_embed.reshape(-1, bs, t, c)[:,:,t//2:t//2+1,:].flatten(1,2)
            mask = mask.reshape(bs, t, -1)[:,t//2:t//2+1,:].flatten(0,1)
            loc_tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
            cls_tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
            if self.more_offset:
                cls_tgt = torch.zeros(num_queries*2, bs, self.d_model, device=refpoint_embed.device)
        else:
            loc_tgt = torch.zeros(num_queries, bs*t, self.d_model, device=refpoint_embed.device)
            cls_tgt = torch.zeros(num_queries, bs*t, self.d_model, device=refpoint_embed.device)
            if self.more_offset:
                cls_tgt = torch.zeros(num_queries*2, bs, self.d_model, device=refpoint_embed.device)
        # tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs*t, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
        # refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_pat*n_q, bs*t, d_model
            # import ipdb; ipdb.set_trace()
        hs, references, cls_hs = self.decoder(cls_tgt, loc_tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references, cls_hs


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

    def __init__(self, decoder_layer, cls_decoder_layer, num_layers, norm=None, cls_norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    offset_embed_diff_each_layer=True,
                    cut_gradient=False,
                    more_offset=False,                              
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.cls_layers = _get_clones(cls_decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.cls_norm = cls_norm
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
        self.cls_ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.more_offset = more_offset
        self.offset_embed = MLP(d_model, d_model, 4, 3)
        if self.more_offset:
            self.offset_embed2 = MLP(d_model, d_model, 4, 3)
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.offset_embed_diff_each_layer = offset_embed_diff_each_layer
        if offset_embed_diff_each_layer:
            self.offset_embed = nn.ModuleList([MLP(d_model, d_model, 4, 3) for i in range(num_layers)])
            if self.more_offset:
                self.offset_embed2 = nn.ModuleList([MLP(d_model, d_model, 4, 3) for i in range(num_layers)])
        self.cls_loc_merger = MLP(2*d_model, d_model, d_model, 2)

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
        self.cut_gradient = cut_gradient

    def forward(self, loc_tgt, cls_tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                ):
        output = loc_tgt
        cls_output = cls_tgt
        if self.more_offset:
            cls_output, cls_output2 = cls_tgt.split(2, dim=0)
            cls_intermediate2 = []
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


            output, dec_attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            
            if layer_id == 0:
                if self.offset_embed_diff_each_layer:
                    offset = self.offset_embed[layer_id](output)
                    if self.more_offset:
                        offset2 = self.offset_embed2[layer_id](output)
                else:
                    offset = self.offset_embed(output)         
                    if self.more_offset:
                        offset2 = self.offset_embed2(output)
            else:
                if self.offset_embed_diff_each_layer:
                    offset = self.offset_embed[layer_id](cls_output)
                    if self.more_offset:
                        offset2 = self.offset_embed2[layer_id](cls_output)
                else:
                    offset = self.offset_embed(cls_output)
                    if self.more_offset:
                        offset2 = self.offset_embed2(cls_output)
            # if layer_id == 0:
            if self.cut_gradient:
                cls_reference_points = (offset + inverse_sigmoid(reference_points.clone().detach())).sigmoid()
                if self.more_offset:
                    cls_reference_points2 = (offset2 + inverse_sigmoid(reference_points.clone().detach())).sigmoid()
            else:
                cls_reference_points = (offset + inverse_sigmoid(reference_points)).sigmoid()
                if self.more_offset:
                    cls_reference_points2 = (offset2 + inverse_sigmoid(reference_points)).sigmoid()
            # else:
                # cls_reference_points = (offset + inverse_sigmoid(cls_reference_points)).sigmoid()
            inter_center = cls_reference_points[..., :self.query_dim]
            cls_query_sine_embed = gen_sineembed_for_position(inter_center)
            cls_query_pos = self.cls_ref_point_head(cls_query_sine_embed) 
            cls_query_sine_embed = cls_query_sine_embed[..., :self.d_model]
            if self.more_offset:
                inter_center2 = cls_reference_points2[..., :self.query_dim]
                cls_query_sine_embed2 = gen_sineembed_for_position(inter_center2)
                cls_query_pos2 = self.cls_ref_point_head(cls_query_sine_embed2) 
                cls_query_sine_embed2 = cls_query_sine_embed2[..., :self.d_model]

            cls_output, cls_dec_attn = cls_layer(cls_output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, cls_query_pos=cls_query_pos, cls_query_sine_embed=cls_query_sine_embed,
                                   is_first=(layer_id == 0))
            if self.more_offset:
                cls_output2, cls_dec_attn2 = cls_layer(cls_output2, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, cls_query_pos=cls_query_pos2, cls_query_sine_embed=cls_query_sine_embed2,
                                    is_first=(layer_id == 0))

            if self.cut_gradient:
                cls_output = self.cls_loc_merger(torch.cat([output.clone().detach(), cls_output], -1))
                if self.more_offset:
                    cls_output2 = self.cls_loc_merger(torch.cat([output.clone().detach(), cls_output2], -1))
            else:
                cls_output = self.cls_loc_merger(torch.cat([output, cls_output], -1))
                if self.more_offset:
                    cls_output2 = self.cls_loc_merger(torch.cat([output, cls_output2], -1))

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
                cls_intermediate.append(self.cls_norm(cls_output))
                if self.more_offset:
                    cls_intermediate2.append(self.cls_norm(cls_output2))

        if self.norm is not None:
            output = self.norm(output)
            cls_output = self.cls_norm(cls_output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                cls_intermediate.pop()
                cls_intermediate.append(cls_output)
                if self.more_offset:
                    cls_intermediate2.pop()
                    cls_intermediate2.append(cls_output2)


        if self.return_intermediate:
            if self.bbox_embed is not None:
                if not self.more_offset:
                    return [
                        torch.stack(intermediate).transpose(1, 2),
                        torch.stack(ref_points).transpose(1, 2),
                        torch.stack(cls_intermediate).transpose(1,2),
                    ]
                else:
                    return [
                        torch.stack(intermediate).transpose(1, 2),
                        torch.stack(ref_points).transpose(1, 2),
                        torch.stack(cls_intermediate).transpose(1,2),
                        torch.stack(cls_intermediate2).transpose(1,2),
                    ]                    
            else:
                if not self.more_offset:
                    return [
                        torch.stack(intermediate).transpose(1, 2), 
                        reference_points.unsqueeze(0).transpose(1, 2),
                        torch.stack(cls_intermediate).transpose(1, 2), 
                    ]
                else:
                    return [
                        torch.stack(intermediate).transpose(1, 2), 
                        reference_points.unsqueeze(0).transpose(1, 2),
                        torch.stack(cls_intermediate).transpose(1, 2),
                        torch.stack(cls_intermediate2).transpose(1, 2), 
                    ]
        if not self.more_offset:
            return output.unsqueeze(0), cls_output.unsqueeze(0)
        else:
            return output.unsqueeze(0), cls_output.unsqueeze(0), cls_output2.unsqueeze(0)


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
                 rm_self_attn_decoder=False):
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
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

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
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

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
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2, dec_attn = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, dec_attn


class TransformerClsDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=True):
        super().__init__()

        # Decoder Cross-Attention
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
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model, stop_middle=False)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder
        self.linear_out = nn.Linear(d_model, d_model)
        constant_(self.linear_out.bias, 0.)
        
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     cls_query_pos = None,
                     cls_query_sine_embed = None,
                     is_first = False,
                     ):

        # ========== Begin of Self-Attention =============        
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(cls_query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(cls_query_pos)
            v = self.sa_v_proj(tgt)

            q = q_content + q_pos # n, bs*t, d
            k = k_content + k_pos # n, bs*t, d

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]

            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(cls_query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        cls_query_sine_embed = self.ca_qpos_sine_proj(cls_query_sine_embed)
        cls_query_sine_embed = cls_query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, cls_query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2, cls_dec_attn = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        # tgt2 = torch.bmm(attn_weights, v.transpose(0,1).contiguous()).transpose(0, 1).contiguous()

        # ========== End of Cross-Attention =============
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, cls_dec_attn


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
        offset_embed_diff_each_layer=cfg.CONFIG.MODEL.OFFSET_EMBED_DIFF_EACH_LAYER,
        use_cls_sa=cfg.CONFIG.MODEL.USE_CLS_SA,
        cut_gradient=cfg.CONFIG.TRAIN.CUT_GRADIENT,
        more_offset=cfg.CONFIG.MODEL.MORE_OFFSET,
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