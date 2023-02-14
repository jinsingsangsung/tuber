# ------------------------------------------------------------------------
# SeqFormer model and criterion classes.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone_3d_builder import build_3d_backbone
from .seqformer.matcher import build_matcher
from .seqformer.segmentation import (SeqFormer, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
                           
from .seqformer.deformable_transformer import build_deformable_transformer
from .criterion import SetCriterionAVA, PostProcessAVA, PosetProcess, MLP
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
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
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        

    def forward(self, samples: NestedTensor, targets, criterion, train=True):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
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
   
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        poses = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n,c,h,w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w)
            # src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
            # mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l+1].shape
            pos_l = pos[l+1].reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
            # pos_l = pos[l+1].reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)
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
                m = samples.mask    # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                mask = mask.unsqueeze(1).repeat(1,samples.tensors.shape[2],1,1) # for ava dataset
                mask = mask.permute(1,0,2,3).flatten(0,1)

                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                src = src.reshape(n//self.num_frames, self.num_frames, c, h, w)
                # src = src.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(1,0,2,3,4)
                mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
                # mask = mask.reshape(n//self.num_frames, self.num_frames, h, w).permute(1,0,2,3)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
                # pos_l = pos_l.reshape(np//self.num_frames, self.num_frames, cp, hp, wp).permute(1,0,2,3,4)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = None
        query_embeds = self.query_embed.weight
        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(srcs, masks, poses, query_embeds)
        # valid_ratios = valid_ratios[: 0]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        # indices_list = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs_box[lvl])
            # tmp = self.bbox_embed[lvl](hs[lvl])
            # tmp.shape: bs, 32, 300, 4
            # reference: bs, 32, 300, 4
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            #TODO: temporally aggregate outputs_coord to output a single frame
            # Now, naively average them
            tmp = tmp.mean(dim=1)

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            # outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # print("outputs_class.shape: ", outputs_class.shape)
        # print("outputs_coord.shape: ", outputs_coord.shape)

        outputs["pred_logits"] = outputs_class[-1]
        outputs["pred_boxes"] = outputs_coord[-1]
        
        if self.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



def build(cfg):    
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(cfg) 
    else:
        backbone = build_backbone(cfg)

    transformer = build_deformable_transformer(cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_frames=cfg.CONFIG.DATA.TEMP_LEN,
        num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
        num_feature_levels=cfg.CONFIG.MODEL.SeqFormer.NUM_FEATURE_LEVELS,
        aux_loss=cfg.CONFIG.TRAIN.AUX_LOSS,
        with_box_refine=cfg.CONFIG.MODEL.SeqFormer.BOX_REFINE,
    )
    
    matcher = build_matcher(cfg)
    
    weight_dict = {'loss_ce': cfg.CONFIG.MATCHER.COST_CLASS, 'loss_bbox': cfg.CONFIG.MATCHER.COST_BBOX}
    weight_dict['loss_giou'] = cfg.CONFIG.MATCHER.COST_GIOU

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    # losses = ['labels', 'cardinality']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterionAVA(weight=10,
                                num_classes=num_classes,
                                num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                matcher=matcher,
                                weight_dict=weight_dict,
                                eos_coef=0.1,
                                losses=losses,
                                data_file="ava",
                                evaluation=cfg.CONFIG.EVAL_ONLY)

    postprocessors = {'bbox': PostProcessAVA() if "ava" in cfg.CONFIG.DATA.DATASET_NAME else PostProcess()}
       
    return model, criterion, postprocessors