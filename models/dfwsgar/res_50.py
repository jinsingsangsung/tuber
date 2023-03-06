# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
# motion augmentation module added


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding


class Backbone(nn.Module):
    def __init__(self, cfg):
        super(Backbone, self).__init__()

        backbone = getattr(torchvision.models, cfg.CONFIG.MODEL.BACKBONE_NAME)(
            replace_stride_with_dilation=[False, False, True], pretrained=False)
        if cfg.CONFIG.MODEL.PRETRAIN_BACKBONE_DIR is not None:
            pretrain_path = cfg.CONFIG.MODEL.PRETRAIN_BACKBONE_DIR
            load_weights(backbone, pretrain_path=pretrain_path)
        self.num_frames = cfg.CONFIG.DATA.TEMP_LEN
        self.num_channels = 2048

        self.motion = True
        self.motion_layer = 4
        self.corr_dim = 64

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if self.motion:
            self.layer_channel = [256, 512, 1024, 2048]

            self.channel_dim = self.layer_channel[self.motion_layer - 1]

            self.corr_input_proj = nn.Sequential(
                nn.Conv2d(self.channel_dim, self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

            self.neighbor_size = cfg.CONFIG.MODEL.NEIGHBOR_SIZE
            self.ps = 2 * cfg.CONFIG.MODEL.NEIGHBOR_SIZE + 1

            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                                 stride=1, padding=0, dilation_patch=1)

            self.corr_output_proj = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.channel_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

    def get_local_corr(self, x):
        x = self.corr_input_proj(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        corr = self.corr_output_proj(corr)

        return corr

    def forward(self, x):
        bs, c, t, h, w = x.shape
        x = x.permute(0,2,1,3,4).contiguous().flatten(0,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer4(x)
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        h_, w_ = x.shape[-2:]
        x = x.view(bs, t, -1, h_, w_).permute(0,2,1,3,4).contiguous()
        return x


class MultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, cfg):
        super(MultiCorrBackbone, self).__init__()

        backbone = getattr(torchvision.models, cfg.CONFIG.MODEL.BACKBONE_NAME)(
            replace_stride_with_dilation=[False, False, True],
            pretrained=True)

        self.num_frames = cfg.CONFIG.DATA.TEMP_LEN
        self.num_channels = 512 if cfg.CONFIG.MODEL.BACKBONE_NAME in ('resnet18', 'resnet34') else 2048

        self.motion = True
        self.motion_layer = 4
        self.corr_dim = 64

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer_channel = [64, 128, 256, 512]

        self.channel_dim = self.layer_channel[self.motion_layer - 1]

        self.corr_input_proj1 = nn.Sequential(
            nn.Conv2d(self.layer_channel[2], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_input_proj2 = nn.Sequential(
            nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.neighbor_size = cfg.CONFIG.MODEL.NEIGHBOR_SIZE
        self.ps = 2 * cfg.CONFIG.MODEL.NEIGHBOR_SIZE + 1

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                             stride=1, padding=0, dilation_patch=1)

        self.corr_output_proj1 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_output_proj2 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False),
            nn.ReLU()
        )

    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        corr = self.get_local_corr(x, 0)
        x = x + corr

        x = self.layer4(x)
        corr = self.get_local_corr(x, 1)
        x = x + corr

        return x


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        features = self[0](x)
        # pos = self[1](features).to(x.dtype)

        return features #, pos



def copy_bn(layer, layer_name, weights, weights_dict_copy, use_affine=True):
    if use_affine:
        copy_weights(layer_weight=layer.weight, weights_name=layer_name + "_s",
                     weights=weights[layer_name + "_s"].reshape(-1, 1, 1, 1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.bias, weights_name=layer_name + "_b",
                     weights=weights[layer_name + "_b"].reshape(-1, 1, 1, 1),
                     weights_dict_copy=weights_dict_copy)
    else:
        copy_weights(layer_weight=layer.weight, weights_name=layer_name + "_s",
                     weights=weights[layer_name + "_s"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.bias, weights_name=layer_name + "_b",
                     weights=weights[layer_name + "_b"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.running_mean, weights_name=layer_name + "_rm",
                     weights=weights[layer_name + "_rm"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.running_var, weights_name=layer_name + "_riv",
                     weights=weights[layer_name + "_riv"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)


def load_weights(model, pretrain_path, load_fc=True, use_affine=False, tune_point=5):
    # import scipy.io as sio
    print("load weights plus")
    r_weights = torch.load(pretrain_path, map_location='cpu')["model"]
    r_weights_copy = torch.load(pretrain_path, map_location='cpu')["model"]
    
    backbone_params = model.state_dict()
    num_backbone_params = len(backbone_params.keys())
    num_updated_params = 0
    not_found_params = []
    for name, p in model.state_dict().items():
        if "module."+name in r_weights.keys():
            backbone_params[name] = r_weights["module."+name]
            num_updated_params += 1
        else:
            not_found_params.append(name)
    print("out of {} parameters, {} parameters are updated".format(num_backbone_params, num_updated_params))
    print("omitted parameters from pretrained backbone model: ", not_found_params)

#     conv1 = model.conv1
#     conv1_bn = model.bn1

#     if tune_point > 1:
#         conv1.weight.requires_grad = False
#         for param in conv1_bn.parameters():
#             param.requires_grad = False

#     res2 = model.layer1
#     res3 = model.layer2
#     res4 = model.layer3
#     res5 = model.layer4

#     stages = [res2, res3, res4, res5]
#     import pdb; pdb.set_trace()
#     copy_weights(layer_weight=conv1.weight, weights_name='conv1.weight', weights=r_weights['conv1.weight'],
#                  weights_dict_copy=r_weights_copy)
#     copy_bn(layer=conv1_bn, layer_name="conv1_spatbn_relu",
#             weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)

#     start_count = [0, 3, 7, 13]
#     import pdb; pdb.set_trace()
#     for s in range(len(stages)):
#         res = stages[s]._modules
#         count = start_count[s]
#         # print(count)
#         for k, block in res.items():
#             # load
#             copy_weights(layer_weight=block.conv1.weight, weights_name="comp_{}_conv_1_w".format(count),
#                          weights=r_weights["comp_{}_conv_1_w".format(count)], weights_dict_copy=r_weights_copy)
#             copy_weights(layer_weight=block.conv3.weight, weights_name="comp_{}_conv_3_w".format(count),
#                          weights=r_weights["comp_{}_conv_3_w".format(count)], weights_dict_copy=r_weights_copy)
#             copy_weights(layer_weight=block.conv4.weight, weights_name="comp_{}_conv_4_w".format(count),
#                          weights=r_weights["comp_{}_conv_4_w".format(count)], weights_dict_copy=r_weights_copy)

#             copy_bn(layer=block.bn1, layer_name="comp_{}_spatbn_1".format(count),
#                     weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)
#             copy_bn(layer=block.bn3, layer_name="comp_{}_spatbn_3".format(count), weights=r_weights,
#                     weights_dict_copy=r_weights_copy, use_affine=use_affine)
#             copy_bn(layer=block.bn4, layer_name="comp_{}_spatbn_4".format(count), weights=r_weights,
#                     weights_dict_copy=r_weights_copy, use_affine=use_affine)

#             if block.down_sample is not None:
#                 down_conv = block.down_sample._modules["0"]
#                 down_bn = block.down_sample._modules["1"]

#                 copy_weights(layer_weight=down_conv.weight, weights_name="shortcut_projection_{}_w".format(count),
#                              weights=r_weights["shortcut_projection_{}_w".format(count)], weights_dict_copy=r_weights_copy)
#                 copy_bn(layer=down_bn, layer_name="shortcut_projection_{}_spatbn".format(count),
#                         weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)

#             count += 1
#         if tune_point > s + 2:
#             for param in stages[s].parameters():
#                 param.requires_grad = False

#     if load_fc:
#         copy_weights(layer_weight=model.out_fc.weight,
#                      weights_name='last_out_L400_w', weights=r_weights['last_out_L400_w'],
#                      weights_dict_copy=r_weights_copy)
#         copy_weights(layer_weight=model.out_fc.bias,
#                      weights_name='last_out_L400_b',
#                      weights=r_weights['last_out_L400_b'][0],
#                      weights_dict_copy=r_weights_copy)

#     print("load pretrain model from " + pretrain_path)
#     print("load fc", load_fc)
#     for k, v in r_weights_copy.items():
#         if "momentum" not in k and "model_iter" not in k and "__globals__" not in k and "__header__" not in k and "lr" not in k and "__version__" not in k:
#             print(k, v.shape)


# def copy_weights(layer_weight, weights_name, weights, weights_dict_copy):
#     assert layer_weight.shape == torch.from_numpy(weights).shape
#     layer_weight.data.copy_(torch.from_numpy(weights))
#     weights_dict_copy.pop(weights_name)


def build_Res50(cfg):
    position_embedding = build_position_encoding(cfg)
    # if cfg.multi_corr:
    #     backbone = MultiCorrBackbone(cfg)
    # else:
    backbone = Backbone(cfg)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
