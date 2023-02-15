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

from models.backbone_3d_builder import build_3d_backbone
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
from models.transformer.transformer import build_transformer
from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder

from models.sparsercnn.detector import SparseRCNN
from models.sparsercnn.loss import SetCriterion, HungarianMatcher
from models.criterion import PostProcess, PostProcessAVA, MLP

def build_model(cfg):
    class_weight = cfg.CONFIG.MODEL.SparseRCNN.CLASS_WEIGHT
    giou_weight = cfg.CONFIG.MODEL.SparseRCNN.GIOU_WEIGHT
    l1_weight = cfg.CONFIG.MODEL.SparseRCNN.L1_WEIGHT
    no_object_weight = cfg.CONFIG.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
    deep_supervision = cfg.CONFIG.MODEL.SparseRCNN.DEEP_SUPERVISION
    use_focal = cfg.CONFIG.MODEL.SparseRCNN.USE_FOCAL
    num_heads = cfg.CONFIG.MODEL.SparseRCNN.NUM_HEADS
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES

    matcher = HungarianMatcher(cfg=cfg,
                               cost_class=class_weight, 
                               cost_bbox=l1_weight, 
                               cost_giou=giou_weight,
                               use_focal=use_focal)

    print('num_classes: ', num_classes)

    backbone = build_3d_backbone(cfg)
    model = SparseRCNN(backbone, cfg)

    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from models.detr.matcher import build_matcher
    else:
        from models.detr.matcher_ucf import build_matcher
    weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
    losses = ['labels', 'boxes']

    criterion = SetCriterion(cfg=cfg,
                            num_classes=num_classes,
                            matcher=matcher,
                            weight_dict=weight_dict,
                            eos_coef=no_object_weight,
                            losses=losses,
                            use_focal=use_focal)

    if cfg.CONFIG.TRAIN.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(num_heads - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] #, 'cardinality'

    postprocessors = {'bbox': PostProcessAVA() if cfg.CONFIG.DATA.DATASET_NAME == 'ava' else PostProcess()}

    return model, criterion, postprocessors
