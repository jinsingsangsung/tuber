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

from models.backbone_builder import build_backbone
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
from models.dab_detr.dab_transformer import build_transformer
from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder
from models.criterion import PostProcess, PostProcessAVA, MLP
from models.criterion import SetCriterion, SetCriterionAVA
from models.transformer.transformer_layers import LSTRTransformerDecoder, LSTRTransformerDecoderLayer, layer_norm
# from models.dn_dab_deformable_detr.dn_components import prepare_for_dn, dn_post_process, compute_dn_loss

import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_frames,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False, two_stage=False, random_refpoints_xy=False, query_dim=4,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava', bbox_embed_diff_each_layer=False, training=True, iter_update=True):
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

        if self.dataset_mode != 'ava':
            self.avg_s = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.query_embed = nn.Embedding(num_queries * temporal_length, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.query_dim = query_dim
        assert query_dim in [2, 4]
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False          

        if "SWIN" in backbone_name:
            print("using swin")
            self.input_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        elif "SlowFast" in backbone_name:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(2048 + 512, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.class_proj = nn.Conv3d(backbone.num_channels[-1], hidden_dim, kernel_size=(4,1,1))
        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)

        if self.dataset_mode == 'ava':
            self.class_embed_b = nn.Linear(hidden_dim, 3)
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed_b = nn.Linear(2048, 2)
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(6)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.iter_update = iter_update
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)

        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.two_stage = two_stage
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride
        self.training = training


    def freeze_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.query_embed.parameters():
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

        features, pos, xt = self.backbone(samples) # both are list of length 4
        src, mask = features[-1].decompose()
        assert mask is not None
        # bs = samples.tensors.shape[0]

        embedweight = self.refpoint_embed.weight      # nq, 4
        hs, reference = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])

        if self.dataset_mode == 'ava':
            outputs_class_b = self.class_embed_b(hs)
        else:
            outputs_class_b = self.class_embed_b(self.avg_s(xt).squeeze(-1).squeeze(-1).squeeze(-1))
            outputs_class_b = outputs_class_b.unsqueeze(0).repeat(6, 1, 1)

        ######## localization head

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

        ############# momentum
        lay_n, bs, nq, dim = hs.shape

        src_c = self.class_proj(xt)        
        hs_t_agg = hs.contiguous().view(lay_n, bs, 1, nq, dim)
        src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1).view(lay_n * bs, self.hidden_dim, -1).permute(2, 0, 1).contiguous()
        if not self.is_swin:
            src_flatten, _ = self.encoder(src_flatten, orig_shape=src_c.shape)

        hs_query = hs_t_agg.view(lay_n * bs, nq, dim).permute(1, 0, 2).contiguous()
        q_class = self.cross_attn(hs_query, src_flatten, src_flatten)[0]
        q_class = q_class.permute(1, 0, 2).contiguous().view(lay_n, bs, nq, self.hidden_dim)

        outputs_class = self.class_embed(self.dropout(q_class))
        # outputs_coord = self.bbox_embed(hs).sigmoid()
    
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
                for a, b ,c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_b[:-1])]


def build_model(cfg):
    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from models.detr.matcher import build_matcher
    else:
        from models.detr.matcher_ucf import build_matcher
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES
    print('num_classes', num_classes)

    backbone = build_backbone(cfg)
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
                 )

    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.CONFIG.LOSS_COFS.DICE_COF, 'loss_bbox': cfg.CONFIG.LOSS_COFS.BBOX_COF}
    weight_dict['loss_giou'] = cfg.CONFIG.LOSS_COFS.GIOU_COF
    weight_dict['loss_ce_b'] = 1
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
    else:
        criterion = SetCriterion(num_classes,
                        matcher=matcher, weight_dict=weight_dict,
                        losses=losses)

    postprocessors = {'bbox': PostProcessAVA() if cfg.CONFIG.DATA.DATASET_NAME == 'ava' else PostProcess()}

    return model, criterion, postprocessors
