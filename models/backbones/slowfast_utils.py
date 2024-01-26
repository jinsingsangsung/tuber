#!/usr/bin/env python3

import torch

def pack_pathway_output(cfg, frames, pathways=2):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `batch` x `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `batch` x `channel` x `num frames` x `height` x `width`.
    """
    if cfg.CONFIG.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[:, [2, 1, 0], :, :, :]
    if pathways==1:
        frame_list = [frames]
    elif pathways==2:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // cfg.CONFIG.SLOWFAST.ALPHA,
                device=frames.device,
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError()
    return frame_list