import sys

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import cv2
from PIL import Image
from torchvision import transforms
from pipelines.video_action_recognition_config import get_cfg_defaults
from models.dab_conv_trans import build_model
# from models.dab_baseline import build_model
from glob import glob
import json
import datasets.video_transforms as T
import random
from thop import profile
from thop import clever_format
import fvcore.nn as fv


cfg = get_cfg_defaults()
cfg.merge_from_file("./configuration/Dab_conv_trans_CSN152_AVA22.yaml")
# cfg.merge_from_file("./configuration/TubeR_CSN50_AVA21.yaml")
model, _, _ = build_model(cfg)

device = "cuda:0"
model = model.to(device)

inp = torch.randn((1, 3, 16, 256, 455)).to(device)
# flops, params = profile(model, (inp, ))
# flops, params = clever_format([flops, params], "%.3f")
# print("flops:", flops)
# print("params:", params)


def get_FLOPs_params(model, model_name, x):
    def _human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
            
    out = fv.FlopCountAnalysis(model, x)
    params = fv.parameter_count(model)
    params_table = fv.parameter_count_table(model)
    flops_human = _human_format(out.total())
    params_human =  _human_format(int(params['']))
    
    return flops_human, params_human

flops, params = get_FLOPs_params(model, "DECSASDE", inp)
print("flops:", flops)
print("params:", params)

# print("=======================Analysis=======================")

# flops = fv.FlopCountAnalysis(model, inp)
# print(fv.flop_count_table(flops))

