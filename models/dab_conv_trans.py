# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
The code refers to https://github.com/facebookresearch/detr
Modified by Zhang Yanyi
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.transformer.util import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from utils.utils import print_log
import os

from models.backbone_3d_builder2 import build_3d_backbone
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
from models.dab_conv_trans_detr.dab_transformer import build_transformer
# from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder
from models.dab_conv_trans_detr.criterion import PostProcess, PostProcessAVA, PostProcessUCF, MLP
from models.dab_conv_trans_detr.criterion import SetCriterion, SetCriterionAVA, SetCriterionUCF, SetCriterionJHMDB
from models.dab_conv_trans_detr.transformer_layers import TransformerEncoderLayer, TransformerEncoder
from models.dab_conv_trans_detr.dab_transformer import TransformerDecoderLayer, TransformerDecoder

import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_frames,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False, two_stage=False, random_refpoints_xy=False, query_dim=4,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava', bbox_embed_diff_each_layer=True, training=True, iter_update=True,
                 gpu_world_rank=0, log_path=None, efficient=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """            
        super(DETR, self).__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.num_frames = num_frames
        self.transformer = transformer
        self.dataset_mode = dataset_mode
        self.num_classes = num_classes
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        
        self.query_dim = query_dim
        assert query_dim in [2, 4]
        self.efficient = efficient
        if not efficient:
            self.refpoint_embed = nn.Embedding(num_queries*temporal_length, 4)
        else:
            assert dataset_mode == "ava", "efficient mode is only for AVA"
            self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.transformer.eff = efficient
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False          

        # if "SWIN" in backbone_name:
        #     if gpu_world_rank == 0: print_log(log_path, "using swin")
        #     self.input_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        #     self.class_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        # elif "SlowFast" in backbone_name:
        #     self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        #     self.class_proj = nn.Conv3d(2048 + 512, hidden_dim, kernel_size=1)
        # else:
            # self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        #     self.class_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        num_feature_levels = 4
        self.num_feature_levels = num_feature_levels
        if not "ViT" in backbone_name:
            if num_feature_levels > 1:            
                self.input_proj = nn.ModuleList()
                num_backbone_outs = len(backbone.strides)
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    self.input_proj.append(nn.Sequential(
                        nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                for _ in range(num_feature_levels - num_backbone_outs):
                    self.input_proj.append(nn.Sequential(
                        nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=(1,2,2), padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                    in_channels = hidden_dim
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )])
            for projection in self.input_proj:                    
                nn.init.xavier_uniform_(projection[0].weight, gain=1)
                nn.init.constant_(projection[0].bias, 0)

        # self.class_proj = nn.Conv3d(backbone.num_channels[-1], hidden_dim, kernel_size=(4,1,1))

        # encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        # self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        # decoder_layer = TransformerDecoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)        
        # decoder_norm = nn.LayerNorm(hidden_dim)
        # self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=3, norm=decoder_norm, return_intermediate=True, query_dim=4, modulate_hw_attn=True, bbox_embed_diff_each_layer=True)
        # self.num_patterns = 3
        # self.num_pattern_message:%3CTQB5fQ7CQ_2uZmev7pIQjA@geopod-ismtpd-5%3Elevel = 4
        # self.patterns = nn.Embedding(self.num_patterns*self.num_pattern_level, hidden_dim)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.dataset_mode == 'ava':
            # self.class_embed = nn.Linear(hidden_dim, num_classes)
            # self.class_embed = nn.Sequential(
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, 1)
            # )
            self.class_embed_b = nn.Linear(hidden_dim, 3)
            # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        else:
            # self.class_embed = nn.Linear(2*hidden_dim, num_classes+1)
            self.class_embed_b = nn.Linear(hidden_dim, 3)
            # self.class_embed.bias.data = torch.ones(num_classes+1) * bias_value
        

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(transformer.num_dec_layers)])
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)


        self.iter_update = iter_update
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed


        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.two_stage = two_stage
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.is_vit = "ViT" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride
        self.training = training


    def freeze_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.class_embed_b.parameters():
            param.requires_grad = False

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples) # both are list of length 4
        srcs = list()
        masks = list()
        poses = list()
        bs = samples.tensors.shape[0]

        if self.is_vit:
            for l, feat in enumerate(features): # 첫 번째 feature는 버림
                src, mask = feat.decompose()
                pos_l = pos[l]
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)     
        else:
            for l, feat in enumerate(features[1:]): # 첫 번째 feature는 버림
                src, mask = feat.decompose()
                src_proj_l = self.input_proj[l](src) # channel 통일
                pos_l = pos[l+1]
                srcs.append(src_proj_l)
                masks.append(mask)
                poses.append(pos_l)

            if self.num_feature_levels > (len(features) - 1): # the last feature map is a projection of the previous map
                _len_srcs = len(features) - 1
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])

                    m = samples.mask    # [B, H, W]
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    mask = mask.unsqueeze(1).expand(-1,src.shape[2],-1,-1)
                    # src: bs, c, t, h, w      mask: bs, t, h, w
                    pos_l = self.backbone[1](NestedTensor(src.flatten(0,1), mask)).to(src.dtype)

                    srcs.append(src)
                    masks.append(mask)
                    poses.append(pos_l)

        assert mask is not None
        # bs = samples.tensors.shape[0]
        if not self.efficient:
            embedweight = self.refpoint_embed.weight.view(self.num_queries, self.temporal_length, 4)      # nq, t, 4
        else:
            embedweight = self.refpoint_embed.weight.view(self.num_queries, 1, 4)

        hs, cls_hs, reference  = self.transformer(srcs, masks, poses, embedweight)
        outputs_class_b = self.class_embed_b(hs)
        ######## localization head
        with torch.autocast("cuda", dtype=torch.float16, enabled=False):
            if not self.bbox_embed_diff_each_layer:
                reference_before_sigmoid = inverse_sigmoid(reference)
                tmp = self.bbox_embed(hs)
                tmp[..., :self.query_dim] += reference_before_sigmoid
                outputs_coord = tmp.sigmoid()
            else:
                reference_before_sigmoid = inverse_sigmoid(reference)
                outputs_coords = []
                for lvl in range(hs.shape[0]):
                    # hs.shape: lay_n, bs, nq, dim
                    tmp = self.bbox_embed[lvl](hs[lvl])
                    tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                    outputs_coord = tmp.sigmoid()
                    outputs_coords.append(outputs_coord)
                outputs_coord = torch.stack(outputs_coords)        

        ######## mix temporal features for classification
        # lay_n, bst, nq, dim = hs.shape
        # hw, bst, ch = memory.shape
        bs = srcs[0].size(0)
        t = self.temporal_length
        # memory = self.encoder(memory, src.shape, mask, pos_embed)
        ##### prepare for the second decoder
        # tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs*t, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
        # embedweight = embedweight.repeat(self.num_patterns, bs, 1) # n_pat*n_q, bst, 4
        # hs_c, ref_c = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=embedweight)
        lay_n = self.transformer.decoder.num_layers
        # outputs_class = self.class_embed(self.dropout(hs)).reshape(lay_n, bs*t, self.num_patterns, self.num_queries, -1).max(dim = 2)[0]
        if not self.efficient:
            outputs_class = self.dropout(cls_hs).mean(dim=-1).reshape(lay_n, bs*t, self.num_queries, -1)
        else:
            outputs_class = self.dropout(cls_hs).mean(dim=-1).reshape(lay_n, bs, self.num_queries, -1)
        if self.dataset_mode == "ava":
            if not self.efficient:
                outputs_class = outputs_class.reshape(-1, bs, t, self.num_queries, self.num_classes)[:,:,self.temporal_length//2,:,:]
                outputs_coord = outputs_coord.reshape(-1, bs, t, self.num_queries, 4)[:,:,self.temporal_length//2,:,:]
                outputs_class_b = outputs_class_b.reshape(-1, bs, t, self.num_queries, 3)[:,:,self.temporal_length//2,:,:]
            else:
                outputs_class = outputs_class.reshape(-1, bs, self.num_queries, self.num_classes)
                outputs_coord = outputs_coord.reshape(-1, bs, self.num_queries, 4)
                outputs_class_b = outputs_class_b.reshape(-1, bs, self.num_queries, 3)
        else:
            outputs_class = outputs_class.reshape(-1, bs, t, self.num_queries, self.num_classes+1)
            outputs_coord = outputs_coord.reshape(-1, bs, t, self.num_queries, 4)
            outputs_class_b = outputs_class_b.reshape(-1, bs, t, self.num_queries, 3)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_b': outputs_class_b[-1],}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_b)
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_b):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_b': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_b[:-1])]


def build_model(cfg):
    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from models.dab_conv_trans_detr.matcher import build_matcher
    elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        from models.dab_conv_trans_detr.matcher_jhmdb import build_matcher
    else:
        from models.dab_conv_trans_detr.matcher_ucf_ import build_matcher
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print_log(log_path, 'num_classes', num_classes)

    backbone = build_3d_backbone(cfg)
    transformer = build_transformer(cfg)

    model = DETR(backbone,
                 transformer,
                 num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                 num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                 num_frames=cfg.CONFIG.MODEL.TEMP_LEN,
                 aux_loss=cfg.CONFIG.TRAIN.AUX_LOSS,
                 hidden_dim=cfg.CONFIG.MODEL.D_MODEL,
                 temporal_length=cfg.CONFIG.MODEL.TEMP_LEN,
                 generate_lfb=cfg.CONFIG.MODEL.GENERATE_LFB,
                 backbone_name=cfg.CONFIG.MODEL.BACKBONE_NAME,
                 ds_rate=cfg.CONFIG.MODEL.DS_RATE,
                 last_stride=cfg.CONFIG.MODEL.LAST_STRIDE,
                 dataset_mode=cfg.CONFIG.DATA.DATASET_NAME,
                 bbox_embed_diff_each_layer=cfg.CONFIG.MODEL.BBOX_EMBED_DIFF_EACH_LAYER,
                 efficient=cfg.CONFIG.EFFICIENT,
                 )

    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.CONFIG.LOSS_COFS.DICE_COF, 'loss_bbox': cfg.CONFIG.LOSS_COFS.BBOX_COF}
    weight_dict['loss_giou'] = cfg.CONFIG.LOSS_COFS.GIOU_COF
    weight_dict['loss_ce_b'] = cfg.CONFIG.LOSS_COFS.PERSON_COF
    # if cfg.CONFIG.MATCHER.BNY_LOSS:
    #     weight_dict['loss_ce_b'] = 1
    #     print("loss binary weight: {}".format(weight_dict['loss_ce_b']))

    if cfg.CONFIG.TRAIN.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.CONFIG.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] #, 'cardinality'

    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        criterion = SetCriterionAVA(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    num_classes,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY)
        postprocessors = {'bbox': PostProcessAVA()}
    elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        criterion = SetCriterionJHMDB(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    num_classes,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,                                    
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY)
        postprocessors = {'bbox': PostProcess()}
    else:
        criterion = SetCriterionUCF(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    num_classes,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,                                    
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY)
        postprocessors = {'bbox': PostProcessUCF()}

    # postprocessors = {'bbox': PostProcessAVA() if cfg.CONFIG.DATA.DATASET_NAME == 'ava' else PostProcess()}

    return model, criterion, postprocessors
