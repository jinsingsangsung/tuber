#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import collections
import torch
import torch.autograd as ag

__all__ = ['prroi_pool2d', '__last', 'box_size', 'concat_shape', 'broadcast', 'meshgrid', 'generate_roi_pool_bins', 'box_intersection', 'generate_intersection_map']


_prroi_pooling = None
COOR_TO_LEN_CORR = 0

def _import_prroi_pooling():
    global _prroi_pooling

    if _prroi_pooling is None:
        try:
            from os.path import join as pjoin, dirname
            from torch.utils.cpp_extension import load as load_extension
            root_dir = pjoin(dirname(__file__), 'src')
            _prroi_pooling = load_extension(
                '_prroi_pooling',
                [pjoin(root_dir, 'prroi_pooling_gpu.c'), pjoin(root_dir, 'prroi_pooling_gpu_impl.cu')],
                verbose=True
            )
        except ImportError:
            raise ImportError('Can not compile Precise RoI Pooling library.')

    return _prroi_pooling


class PrRoIPool2DFunction(ag.Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        _prroi_pooling = _import_prroi_pooling()

        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), \
                'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())

        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)

        features = features.contiguous()
        rois = rois.contiguous()
        params = (pooled_height, pooled_width, spatial_scale)

        if features.is_cuda:
            output = _prroi_pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            # everything here is contiguous.
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        _prroi_pooling = _import_prroi_pooling()

        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None

        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = _prroi_pooling.prroi_pooling_backward_cuda(features, rois, output, grad_output, *ctx.params)
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = _prroi_pooling.prroi_pooling_coor_backward_cuda(features, rois, output, grad_output, *ctx.params)

        return grad_input, grad_coor, None, None, None


prroi_pool2d = PrRoIPool2DFunction.apply

def __last(arr, x):
    return arr.narrow(-1, x, 1).squeeze(-1)

def box_size(box, c2l=COOR_TO_LEN_CORR):
    return (__last(box, 2) - __last(box, 0) + c2l) * (__last(box, 3) - __last(box, 1) + c2l)    

def concat_shape(*shapes):
    """Concatenate shapes into a tuple. The values can be either torch.Size, tuple, list, or int."""
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)

def broadcast(tensor, dim, size):
    """Broadcast a specific dim for `size` times. Originally the dim size must be 1."""
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim+1:]))


def meshgrid(input1, input2=None, dim=-1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def generate_roi_pool_bins(box, bin_size, c2l=COOR_TO_LEN_CORR):
    # TODO(Jiayuan Mao @ 07/20): workaround: line space is not implemented for cuda.
    linspace = torch.linspace(0, 1, bin_size + 1, dtype=box.dtype).to(device=box.device)
    for i in range(box.dim() - 1):
        linspace.unsqueeze_(0)
    x_space = linspace * (__last(box, 2) - __last(box, 0) + c2l).unsqueeze(-1) + __last(box, 0).unsqueeze(-1)
    y_space = linspace * (__last(box, 3) - __last(box, 1) + c2l).unsqueeze(-1) + __last(box, 1).unsqueeze(-1)
    x1, x2 = x_space[:, :-1], x_space[:, 1:] - c2l
    y1, y2 = y_space[:, :-1], y_space[:, 1:] - c2l
    y1, x1 = meshgrid(y1, x1, dim=-1)
    y2, x2 = meshgrid(y2, x2, dim=-1)

    # shape: nr_boxes, bin_size^2, 4
    bins = torch.stack([x1, y1, x2, y2], dim=-1).view(box.size(0), -1, 4)
    return bins.float()

def box_intersection(box1, box2, ratio=False, c2l=COOR_TO_LEN_CORR):
    xmin, ymin = [torch.max(__last(box1, i), __last(box2, i)) for i in range(2)]
    xmax, ymax = [torch.min(__last(box1, i), __last(box2, i)) for i in range(2, 4)]
    iw = torch.max(xmax - xmin + c2l, torch.zeros_like(xmax))
    ih = torch.max(ymax - ymin + c2l, torch.zeros_like(ymax))
    inter = iw * ih
    if ratio:
        return inter / box_size(box2)
    return inter

def generate_intersection_map(box1, box2, bin_size, c2l=COOR_TO_LEN_CORR):
    # box: nr_boxes, 4
    # bins: nr_boxes, bin_size^2, 4
    bins = generate_roi_pool_bins(box2, bin_size, c2l)
    box1 = box1.unsqueeze(1).expand_as(bins)
    return box_intersection(box1, bins, ratio=True, c2l=c2l).view(box1.size(0), 1, bin_size, bin_size).float()