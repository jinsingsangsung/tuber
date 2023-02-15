#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from detectron2.layers import ShapeSpec

__all__ = ["SparseRCNN"]


@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, backbone, cfg):
        super().__init__()

        # self.device = torch.device(cfg.CONFIG.MODEL.DEVICE)

        self.in_features = cfg.CONFIG.MODEL.SparseRCNN.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.CONFIG.DATA.NUM_CLASSES
        self.num_proposals = cfg.CONFIG.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.CONFIG.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.CONFIG.MODEL.SparseRCNN.NUM_HEADS
        self.num_frames = cfg.CONFIG.MODEL.TEMP_LEN

        # Build Backbone.
        self.backbone = backbone
        # self.size_divisibility = self.backbone.size_divisibility
        
        # Feature dimension matcher
        num_feature_levels = len(self.in_features)
        hidden_dim = self.hidden_dim
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.output_shape = {
            name: (ShapeSpec(
                channels=hidden_dim, stride=backbone.strides[l]
            ) if (l < len(backbone.num_channels))
                else ShapeSpec(
                    channels=hidden_dim, stride=backbone.strides[-1]*2
                ))
            for l, name in enumerate(self.in_features)
        }
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.output_shape)

        # Loss parameters:
        class_weight = cfg.CONFIG.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.CONFIG.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.CONFIG.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.CONFIG.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.CONFIG.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.CONFIG.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        # matcher = HungarianMatcher(cfg=cfg,
        #                            cost_class=class_weight, 
        #                            cost_bbox=l1_weight, 
        #                            cost_giou=giou_weight,
        #                            use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]



        # pixel_mean = torch.Tensor(cfg.CONFIG.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        # pixel_std = torch.Tensor(cfg.CONFIG.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        # self.to(self.device)


    def forward(self, images: NestedTensor):

        # images, images_whwh = self.preprocess_image(batched_inputs)
        if not isinstance(images, NestedTensor):
            images = nested_tensor_from_tensor_list(images)
        # print("images.tensors.shape: ", images.tensors.shape)
        # Feature Extraction.
        features, _ = self.backbone(images)
        images_whwh = self.images_whwh_creator(images)
        srcs = list()

        for l, feat in enumerate(features[1:]):
            src, _ = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n,c,h,w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w)
            # TODO: how to pool temporal frames? Naively, take max as of now
            src_proj_l = src_proj_l.max(dim=1).values
            srcs.append(src_proj_l)
        num_feature_levels = len(self.in_features) # 4
        if num_feature_levels > (len(features) - 1): # the last feature map is a projection of the previous map
            _len_srcs = len(features) - 1 # 2
            for l in range(_len_srcs, num_feature_levels):
                if l == _len_srcs:
                    # print("l:", l)
                    # print("len(self.input_proj):", len(self.input_proj))
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                srcs.append(src)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(srcs, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        return output
        # if self.training:
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        #     targets = self.prepare_targets(gt_instances)
        #     if self.deep_supervision:
        #         output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
        #                                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        #     loss_dict = self.criterion(output, targets)
        #     weight_dict = self.criterion.weight_dict
        #     for k in loss_dict.keys():
        #         if k in weight_dict:
        #             loss_dict[k] *= weight_dict[k]
        #     return loss_dict

        # else:
        #     box_cls = output["pred_logits"]
        #     box_pred = output["pred_boxes"]
        #     results = self.inference(box_cls, box_pred, images.image_sizes)
            
            # return results
            

    # def prepare_targets(self, targets):
    #     new_targets = []
    #     for targets_per_image in targets:
    #         target = {}
    #         h, w = targets_per_image.image_size
    #         image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
    #         gt_classes = targets_per_image.gt_classes
    #         gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
    #         gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
    #         target["labels"] = gt_classes.to(self.device)
    #         target["boxes"] = gt_boxes.to(self.device)
    #         target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
    #         target["image_size_xyxy"] = image_size_xyxy.to(self.device)
    #         image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
    #         target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
    #         target["area"] = targets_per_image.gt_boxes.area().to(self.device)
    #         new_targets.append(target)

    #     return new_targets

    # def inference(self, box_cls, box_pred, image_sizes):
    #     """
    #     Arguments:
    #         box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
    #             The tensor predicts the classification probability for each proposal.
    #         box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
    #             The tensor predicts 4-vector (x,y,w,h) box
    #             regression values for every proposal
    #         image_sizes (List[torch.Size]): the input image sizes

    #     Returns:
    #         results (List[Instances]): a list of #images elements.
    #     """
    #     assert len(box_cls) == len(image_sizes)
    #     results = []

    #     if self.use_focal:
    #         scores = torch.sigmoid(box_cls)
    #         labels = torch.arange(self.num_classes, device=self.device).\
    #                  unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

    #         for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
    #                 scores, box_pred, image_sizes
    #         )):
    #             result = Instances(image_size)
    #             scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
    #             labels_per_image = labels[topk_indices]
    #             box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
    #             box_pred_per_image = box_pred_per_image[topk_indices]

    #             result.pred_boxes = Boxes(box_pred_per_image)
    #             result.scores = scores_per_image
    #             result.pred_classes = labels_per_image
    #             results.append(result)

    #     else:
    #         # For each box we assign the best class or the second best if the best on is `no_object`.
    #         scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

    #         for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
    #             scores, labels, box_pred, image_sizes
    #         )):
    #             result = Instances(image_size)
    #             result.pred_boxes = Boxes(box_pred_per_image)
    #             result.scores = scores_per_image
    #             result.pred_classes = labels_per_image
    #             results.append(result)

    #     return results

    # def preprocess_image(self, batched_inputs):
    #     """
    #     Normalize, pad and batch the input images.
    #     """
    #     # images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
    #     images = x["image"].to(self.device) for x in batched_inputs]
    #     images = ImageList.from_tensors(images, self.size_divisibility)

    #     images_whwh = list()
    #     for bi in batched_inputs:
    #         h, w = bi["image"].shape[-2:]
    #         images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
    #     images_whwh = torch.stack(images_whwh)

    #     return images, images_whwh
    
    def images_whwh_creator(self, images: NestedTensor):
        batched_image = images.tensors
        bs, c, t, h, w = batched_image.shape
        images_whwh = list()
        for bi in batched_image:
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=batched_image.device))
        images_whwh = torch.stack(images_whwh)
        return images_whwh

