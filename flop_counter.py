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
from models.dab_hier import build_model
from glob import glob
import json
import datasets.video_transforms as T
import random
from thop import profile

cfg = get_cfg_defaults()
cfg.merge_from_file("./configuration/Dab_hier_CSN152_AVA22.yaml")
model, _, _ = build_model(cfg)

device = "cuda:0"
model = model.to(device)

inp = torch.randn((1, 3, 32, 256, 455)).to(device)
flops, params = profile(model, (inp, ))
print("gflops:", flops / 1000**3)
print("params:", params / 10**6)
