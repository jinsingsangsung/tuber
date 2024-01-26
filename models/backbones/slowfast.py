# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

from functools import partial
import torch
import torch.nn as nn
from .sfmodels import resnet_helper, stem_helper  # noqa
from utils.misc import NestedTensor
from models.dab_conv_trans_detr.position_encoding import build_position_encoding
import torch.nn.functional as F
import os
from utils.utils import print_log
import pickle
from utils.c2_model_loading import load_c2_format


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = nn.BatchNorm3d
        self.cfg = cfg
        self.num_pathways = 2
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        pool_size = _POOL1['slowfast']
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.CONFIG.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.CONFIG.RESNET.DEPTH]

        self.alpha = cfg.CONFIG.SLOWFAST.ALPHA
        num_groups = cfg.CONFIG.RESNET.NUM_GROUPS
        width_per_group = cfg.CONFIG.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.CONFIG.SLOWFAST.BETA_INV // cfg.CONFIG.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS['slowfast']

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.CONFIG.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.CONFIG.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.CONFIG.SLOWFAST.BETA_INV,
            cfg.CONFIG.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.CONFIG.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.CONFIG.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.CONFIG.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.CONFIG.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.CONFIG.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.CONFIG.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.CONFIG.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.CONFIG.NONLOCAL.POOL[0],
            instantiation=cfg.CONFIG.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.CONFIG.RESNET.TRANS_FUNC,
            dilation=cfg.CONFIG.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.CONFIG.SLOWFAST.BETA_INV,
            cfg.CONFIG.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.CONFIG.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.CONFIG.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.CONFIG.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.CONFIG.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.CONFIG.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.CONFIG.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.CONFIG.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.CONFIG.NONLOCAL.POOL[1],
            instantiation=cfg.CONFIG.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.CONFIG.RESNET.TRANS_FUNC,
            dilation=cfg.CONFIG.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.CONFIG.SLOWFAST.BETA_INV,
            cfg.CONFIG.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.CONFIG.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.CONFIG.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.CONFIG.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.CONFIG.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.CONFIG.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.CONFIG.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.CONFIG.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.CONFIG.NONLOCAL.POOL[2],
            instantiation=cfg.CONFIG.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.CONFIG.RESNET.TRANS_FUNC,
            dilation=cfg.CONFIG.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.CONFIG.SLOWFAST.BETA_INV,
            cfg.CONFIG.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.CONFIG.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.CONFIG.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.CONFIG.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.CONFIG.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.CONFIG.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.CONFIG.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.CONFIG.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.CONFIG.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.CONFIG.NONLOCAL.POOL[3],
            instantiation=cfg.CONFIG.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.CONFIG.RESNET.TRANS_FUNC,
            dilation=cfg.CONFIG.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

    def _init_position_encoding(self, cfg):
        self.position_encoding = build_position_encoding(cfg.MODEL.STM.HIDDEN_DIM)
        
    def forward(self, x):
        inputs = []
        use_mask = isinstance(x[0], NestedTensor)
        if use_mask:
            self._init_position_encoding(self.cfg)
            tensors = []
            masks = []
            for x_ in x:
                tensors.append(x_.tensors)
                masks.append(x_.mask)
            x = tensors[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        fm = torch.cat([x[0], F.max_pool3d(x[1], kernel_size=(self.alpha, 1, 1), stride=(self.alpha, 1, 1))], dim=1)
        inputs.append(fm)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        fm = torch.cat([x[0], F.max_pool3d(x[1], kernel_size=(self.alpha, 1, 1), stride=(self.alpha, 1, 1))], dim=1)
        inputs.append(fm)
        x = self.s3_fuse(x)
        x = self.s4(x)
        fm = torch.cat([x[0], F.max_pool3d(x[1], kernel_size=(self.alpha, 1, 1), stride=(self.alpha, 1, 1))], dim=1)
        inputs.append(fm)
        x = self.s4_fuse(x)
        x = self.s5(x)
        fm = torch.cat([x[0], F.max_pool3d(x[1], kernel_size=(self.alpha, 1, 1), stride=(self.alpha, 1, 1))], dim=1)
        inputs.append(fm)
        if use_mask:
            out = []
            pos = []
            for input in inputs:
                m = masks[0]
                assert m is not None
                mask = F.interpolate(m[None].float(), size=input.shape[-2:]).to(torch.bool)[0]
                mask = mask.unsqueeze(1).repeat(1,input.shape[2],1,1)
                out.append(NestedTensor(input, mask))
                pos.append(self.position_encoding(NestedTensor(input, mask)))
            return out, pos
        return inputs

def load_weights(model, pretrain_path, load_fc, tune_point, gpu_world_rank, log_path):
    checkpoint = load_c2_format(pretrain_path)

    model_dict = model.state_dict()
    # print("not found layers: ", len([k for k in checkpoint.keys() if k not in model_dict.keys()]))
    # print("not found layers: ", [k for k in checkpoint.keys() if k not in model_dict.keys()])
    # import pdb; pdb.set_trace()
    not_found_layers = [k for k in checkpoint.keys() if k not in model_dict.keys()]
    for layer in not_found_layers:
        checkpoint.pop(layer)

    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)

    if gpu_world_rank == 0:
        print_log(log_path, f"pretrained backbone weights loaded from: {pretrain_path}")
    
    
def build_SlowFast(cfg):
    model = SlowFast(cfg)
    tune_point = cfg.CONFIG.MODEL.TUNE_POINT
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if cfg.CONFIG.MODEL.PRETRAINED:
        load_weights(model,
                     pretrain_path=cfg.CONFIG.MODEL.PRETRAIN_BACKBONE_DIR,
                     load_fc=False,
                     tune_point=tune_point,
                     gpu_world_rank=cfg.DDP_CONFIG.GPU_WORLD_RANK,
                     log_path=log_path,
                     )
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print_log(log_path, "build SlowFast, tune point: {}".format(tune_point))
    return model