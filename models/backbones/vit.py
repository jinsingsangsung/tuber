# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .vit_utils import PatchEmbed, get_sinusoid_encoding_table, Block, interpolate_pos_embed_online
import os
from utils.utils import print_log


class ViT(nn.Module):
    def __init__(self, cfg):
        super(ViT, self).__init__()
        self.num_pathways = 1
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway Swin model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # default cfg for vit-base
        tubelet_size = cfg.CONFIG.ViT.TUBELET_SIZE
        patch_size = cfg.CONFIG.ViT.PATCH_SIZE
        in_chans = cfg.CONFIG.ViT.IN_CHANS
        embed_dim = cfg.CONFIG.ViT.EMBED_DIM
        pretrain_img_size = cfg.CONFIG.ViT.PRETRAIN_IMG_SIZE
        use_learnable_pos_emb = cfg.CONFIG.ViT.USE_LEARNABLE_POS_EMB
        drop_rate = cfg.CONFIG.ViT.DROP_RATE
        attn_drop_rate = cfg.CONFIG.ViT.ATTN_DROP_RATE
        drop_path_rate = cfg.CONFIG.ViT.DROP_PATH_RATE
        depth = cfg.CONFIG.ViT.DEPTH
        num_heads = cfg.CONFIG.ViT.NUM_HEADS
        mlp_ratio = cfg.CONFIG.ViT.MLP_RATIO
        qkv_bias = cfg.CONFIG.ViT.QKV_BIAS
        qk_scale = cfg.CONFIG.ViT.QK_SCALE
        init_values = cfg.CONFIG.ViT.INIT_VALUES
        use_checkpoint = cfg.CONFIG.ViT.USE_CHECKPOINT
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.depth = depth  # 12
        self.tubelet_size = tubelet_size

        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim, tubelet_size=self.tubelet_size, num_frames=cfg.CONFIG.DATA.TEMP_LEN,)
        num_patches = self.patch_embed.num_patches  # 8x14x14
        self.grid_size = [pretrain_img_size // patch_size, pretrain_img_size // patch_size]  # [14,14]
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_checkpoint=use_checkpoint)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


    def forward(self, x_list):

        x = x_list
        # print(x.shape) b, 3, 16, 244, 244
        x = self.patch_embed(x)  # x.shape=[b 768 8 14 14]
        ws = x.shape[2:]  # t,h,w
        num_frame = x.shape[2]
        assert num_frame % 2 == 0, "Only consider even case, check frames {}".format(num_frame)

        x = x.flatten(2).transpose(1, 2).contiguous()  # b,thw,768
        B, _, C = x.shape
        pos_embed = self.pos_embed
        if self.pos_embed.shape[1] != ws[0] * ws[1] * ws[2]:
            # pos_embed=[1 8*14*14 384]->[1 8*16*29 384]
            pos_embed = pos_embed.reshape(ws[0], -1, C)
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [ws[1], ws[2]], 0).reshape(1, -1, C)

        x = x + pos_embed.type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        for i in range(self.depth):
            blk = self.blocks[i]
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        # b,thw,768->b,768,t,h,w
        x = x.reshape(B, ws[0], ws[1], ws[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        features = [x, x, x, x]
        return features


def load_weights(model, pretrain_path, load_fc, tune_point, gpu_world_rank, log_path):
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    model_dict = model.state_dict()
    # print([k for k in checkpoint["module"].keys() if k not in model_dict.keys()])
    if not load_fc:
        fc_layers = ["fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"]
        for layer in fc_layers:
            checkpoint["module"].pop(layer)

    model_dict.update(checkpoint["module"])
    model.load_state_dict(model_dict)
    if tune_point > 0:
        for name, param in model.named_parameters(): # default: weights are all frozen
            if "patch_embed" in name:
                param.requires_grad_(False)
            elif "norm" in name:
                param.requires_grad_(False)
            elif int(name.split(".")[1]) <= tune_point:
                param.requires_grad_(False)
            else:
                pass
    if gpu_world_rank == 0:
        print_log(log_path, f"pretrained backbone weights loaded from: {pretrain_path}")
    
    
def build_ViT(cfg):
    model = ViT(cfg)
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
        print_log(log_path, "build ViT, tune point: {}".format(tune_point))
    return model