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
                       is_dist_avail_and_initialized)

from models.backbone_builder import build_backbone
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
from models.transformer.transformer import build_transformer
from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder
from models.criterion import SetCriterion, PostProcess, SetCriterionAVA, PostProcessAVA, MLP

from .prroi_pool import PrRoIPool2D
from .functional import generate_intersection_map
from models.transformer.position_encoding import build_position_encoding
import copy

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.transformer = transformer
        self.avg = nn.AvgPool3d(kernel_size=(temporal_length, 1, 1))
        self.dataset_mode = dataset_mode

        if self.dataset_mode != 'ava':
            self.avg_s = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.query_embed = nn.Embedding(num_queries * temporal_length, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            
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

        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)

        if self.dataset_mode == 'ava':
            self.class_embed_b = nn.Linear(hidden_dim, 3)
        else:
            self.class_embed_b = nn.Linear(2048, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.dataset_mode == 'ava':
            self.class_fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_fc = nn.Linear(hidden_dim, num_classes + 1)
        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride

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

    def forward(self, samples: NestedTensor):
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

        features, pos, xt = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        if self.dataset_mode == 'ava':
            outputs_class_b = self.class_embed_b(hs)
        else:
            outputs_class_b = self.class_embed_b(self.avg_s(xt).squeeze(-1).squeeze(-1).squeeze(-1))
            outputs_class_b = outputs_class_b.unsqueeze(0).repeat(6, 1, 1)
        #############momentum
        lay_n, bs, nb, dim = hs.shape

        src_c = self.class_proj(xt)

        hs_t_agg = hs.contiguous().view(lay_n, bs, 1, nb, dim)

        src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1).view(lay_n * bs, self.hidden_dim, -1).permute(2, 0, 1).contiguous()
        if not self.is_swin:
            src_flatten, _ = self.encoder(src_flatten, orig_shape=src_c.shape)

        hs_query = hs_t_agg.view(lay_n * bs, nb, dim).permute(1, 0, 2).contiguous()
        q_class = self.cross_attn(hs_query, src_flatten, src_flatten)[0]
        q_class = q_class.permute(1, 0, 2).contiguous().view(lay_n, bs, nb, self.hidden_dim)

        outputs_class = self.class_fc(self.dropout(q_class))
        outputs_coord = self.bbox_embed(hs).sigmoid()

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

class DETR_GT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True, dataset_mode='ava', context_concat=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.transformer = transformer
        self.avg = nn.AvgPool3d(kernel_size=(temporal_length, 1, 1))
        self.dataset_mode = dataset_mode

        if self.dataset_mode != 'ava':
            self.avg_s = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.query_embed = nn.Embedding(num_queries * temporal_length, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            
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

        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)

        if self.dataset_mode == 'ava':
            self.class_embed_b = nn.Linear(hidden_dim, 3)
        else:
            self.class_embed_b = nn.Linear(2048, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.dataset_mode == 'ava':
            self.class_fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_fc = nn.Linear(hidden_dim, num_classes + 1)
        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name
        self.generate_lfb = generate_lfb
        self.last_stride = last_stride

        self.pool_size = 7 # add this to cfg
        self.downsample_rate = 16 # add this to cfg
        self.object_roi_pool = PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / self.downsample_rate)
        self.context_roi_pool = PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / self.downsample_rate)       
        self.context_concat = context_concat
        if context_concat:
            self.object_feature_fuse = nn.Conv2d(2048 * 2, 2048, 1)
            self.context_feature_extract = nn.Conv2d(2048, 2048, 1)
        self.position_encoding = build_position_encoding(hidden_dim)
        self.num_classes = num_classes
        

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

    def forward(self, targets, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

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

        features, pos, xt = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        ####### put GT boxes here, extract features from src
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # tgt_bbox = tgt_bbox[:,1:]
        
        # below is adapted from https://github.com/vacancy/NSCL-PyTorch-Release/blob/ef493d58a986dcb1ea16a23183a57119abeaff34/nscl/nn/scene_graph/scene_graph.py
        bs, _, t, w, h = src.shape # note that if cfg.MODEL.SINGLE_FRAME: t is downsampled to 1 (case of AVA)
        # src_context = copy.deepcopy(src)
        src = src.contiguous().permute(0,2,1,3,4).flatten(0,1)

        num_boxes_per_batch_idx = []
        this_object_features_list = []
        tgt_bboxes_list = []

        for i, v in enumerate(targets): # i iterates over batch size
            with torch.no_grad():
                num_boxes = len(v["boxes"])
                num_boxes_per_batch_idx.append(num_boxes)
                ## TODO: what happens if num_boxes == 0?
                if num_boxes == 0:
                    pass
                tgt_bbox = v["boxes"][:, 1:] # num_boxes x 4
                tgt_bboxes_list.append(tgt_bbox)
                batch_ind = i + torch.zeros(num_boxes, 1, dtype=tgt_bbox.dtype, device=tgt_bbox.device)
                image_h, image_w = h * self.downsample_rate, w * self.downsample_rate
                if self.context_concat:
                    context_features = self.context_feature_extract(src)
                    image_box = torch.cat([
                        torch.zeros(num_boxes, 1, dtype=tgt_bbox.dtype, device=tgt_bbox.device),
                        torch.zeros(num_boxes, 1, dtype=tgt_bbox.dtype, device=tgt_bbox.device),
                        image_w + torch.zeros(num_boxes, 1, dtype=tgt_bbox.dtype, device=tgt_bbox.device),
                        image_h + torch.zeros(num_boxes, 1, dtype=tgt_bbox.dtype, device=tgt_bbox.device)
                    ], dim=-1)
                    box_context_imap = generate_intersection_map(tgt_bbox, image_box, self.pool_size)
                    this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
                    x, y = this_context_features.chunk(2, dim=1)
                    this_object_features = self.object_feature_fuse(
                        torch.cat([
                        self.object_roi_pool(
                            features = src,
                            rois = torch.cat([batch_ind, tgt_bbox], dim=-1)
                        ),
                        x, y * box_context_imap
                        ], dim=1)
                    )
                else:
                    this_object_features = self.object_roi_pool(
                                                features = src,
                                                rois = torch.cat([batch_ind, tgt_bbox], dim=-1)
                                            )
                this_object_features_list.append(this_object_features)
        input_object_features = torch.cat(this_object_features_list, dim=0).unsqueeze(2)
        tgt_bboxes = torch.cat(tgt_bboxes_list)
        # print("input_object_features.shape: ", input_object_features.shape)
        # print("input_object_features: ", input_object_features[0, 0,:,:])

        assert sum(num_boxes_per_batch_idx) == len(input_object_features)
        if sum(num_boxes_per_batch_idx) == 0: #only works for ava
            out = {'pred_logits': torch.zeros(bs, self.num_queries, self.num_classes, device=tgt_bbox.device),
                   'pred_boxes': torch.zeros(bs, self.num_queries, 4, device=tgt_bbox.device),
                   'pred_logits_b': torch.zeros(bs, self.num_queries, 3, device=tgt_bbox.device),
                   }
            return out
        # print(len(targets))
        # print(tgt_bbox.shape, features[0].tensors.shape)
        mask = torch.ones(input_object_features.size(0), 1, self.pool_size, self.pool_size, device=input_object_features.device)<1
        pos = self.position_encoding(NestedTensor(input_object_features, mask))
        hs = self.transformer(self.input_proj(input_object_features), mask, self.query_embed.weight, pos)[0] # layer_num, num_boxes(sum over all batch), num_query, hidden_dim
        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        if self.dataset_mode == 'ava':
            outputs_class_b = self.class_embed_b(hs)
        else:
            outputs_class_b = self.class_embed_b(self.avg_s(xt).squeeze(-1).squeeze(-1).squeeze(-1))
            outputs_class_b = outputs_class_b.unsqueeze(0).repeat(6, 1, 1)
        ############# momentum
        lay_n, boxes, nb, dim = hs.shape #bs: no longer a batch size;--> boxes

        src_c = self.class_proj(xt)

        hs_t_agg = hs.view(lay_n, boxes, 1, nb, dim).contiguous()
        # src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1).view(lay_n * bs, self.hidden_dim, -1).permute(2, 0, 1).contiguous()
        src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1)
        src_flatten_ = list(src_flatten.chunk(bs, dim=1)) # 6, bs, 256, -1
        for i, num_boxes in enumerate(num_boxes_per_batch_idx):
            src_flatten_[i] = src_flatten_[i].repeat(1, num_boxes, 1, 1)
        src_flatten = torch.cat(src_flatten_, dim=1).view(lay_n * boxes, self.hidden_dim, -1).permute(2, 0, 1).contiguous()

        if not self.is_swin:
            src_flatten, _ = self.encoder(src_flatten, orig_shape=src_c.shape)

        hs_query = hs_t_agg.view(lay_n * boxes, nb, dim).permute(1, 0, 2).contiguous()
        q_class = self.cross_attn(hs_query, src_flatten, src_flatten)[0]
        q_class = q_class.permute(1, 0, 2).contiguous().view(lay_n, boxes, nb, self.hidden_dim)

        outputs_class = self.class_fc(self.dropout(q_class))
        # outputs_coord = self.bbox_embed(hs).sigmoid()
        # outputs_coord = tgt_bboxes.repeat(nb, 1).view(boxes, nb, 4).unsqueeze(0) #need to check the order of the axis
        outputs_coord = tgt_bboxes.unsqueeze(1).repeat(1, nb, 1).view(boxes, nb, 4).unsqueeze(0)
        # print(outputs_coord.shape)
        # print(outputs_coord[0, :, :3, :])

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_b': outputs_class_b[-1],}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_b)

        return out, num_boxes_per_batch_idx
    
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

    model = DETR_GT(backbone,
                 transformer,
                 num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                 num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                 aux_loss=cfg.CONFIG.TRAIN.AUX_LOSS,
                 hidden_dim=cfg.CONFIG.MODEL.D_MODEL,
                 temporal_length=cfg.CONFIG.MODEL.TEMP_LEN,
                 generate_lfb=cfg.CONFIG.MODEL.GENERATE_LFB,
                 backbone_name=cfg.CONFIG.MODEL.BACKBONE_NAME,
                 ds_rate=cfg.CONFIG.MODEL.DS_RATE,
                 last_stride=cfg.CONFIG.MODEL.LAST_STRIDE,
                 dataset_mode=cfg.CONFIG.DATA.DATASET_NAME)

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
        criterion = SetCriterion(cfg.CONFIG.LOSS_COFS.WEIGHT,
                        num_classes,
                        num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                        matcher=matcher, weight_dict=weight_dict,
                        eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                        losses=losses,
                        data_file=cfg.CONFIG.DATA.DATASET_NAME,
                        evaluation=cfg.CONFIG.EVAL_ONLY)

    postprocessors = {'bbox': PostProcessAVA() if cfg.CONFIG.DATA.DATASET_NAME == 'ava' else PostProcess()}

    return model, criterion, postprocessors
