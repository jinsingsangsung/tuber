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

def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id:" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None
            items[81] = "happens"

    return items

items = read_label_map("../assets/ava_action_list_v2.1.pbtxt")

def make_transforms(image_set, cfg):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("transform image crop: {}".format(cfg.CONFIG.DATA.IMG_SIZE))
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            T.ColorJitter(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])

    if image_set == 'visual':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

cfg = get_cfg_defaults()
cfg.merge_from_file("./configuration/Dab_hier_CSN50_AVA22.yaml")
model, _, _ = build_model(cfg)

checkpoint = torch.load("../pretrained_models/main/dab_hier.pth")
model_dict = model.state_dict()
pretrained_dict = {k[7:]: v for k, v in checkpoint['model'].items() if k[7:] in model_dict}
unused_dict = {k[:7]: v for k, v in checkpoint['model'].items() if not k[7:] in model_dict}
not_found_dict = {k: v for k, v in model_dict.items() if not "module."+k in checkpoint['model']}
print("# successfully loaded model layers:", len(pretrained_dict.keys()))
print("# unused model layers:", len(unused_dict.keys()))
print("# not found layers:", len(not_found_dict.keys()))
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

transforms=make_transforms("val", cfg)

model.eval()
sample_image1_path = "/mnt/tmp/frames/xeGWXqSvC-8/xeGWXqSvC-8_000360.jpg" #False
sample_image2_path = "/mnt/tmp/frames/CMCPhm2L400/CMCPhm2L400_011200.jpg" #False
sample_image3_path = "/mnt/tmp/frames/Gvp-cj3bmIY/Gvp-cj3bmIY_024750.jpg" #True

# '/home/nsml/assets/ava_{}_v21.json'
val_bbox_json = json.load(open(cfg.CONFIG.DATA.ANNO_PATH.format("val")))
video_frame_bbox = val_bbox_json["video_frame_bbox"]


def load_annotation(sample_id, video_frame_list): # (val 혹은 train의 key frame을 표시해놓은 list)

    num_classes = 80
    boxes, classes = [], []
    target = {}

    first_img = cv2.imread(video_frame_list[0])

    oh = first_img.shape[0]
    ow = first_img.shape[1]
    if oh <= ow:
        nh = 256
        nw = 256 * (ow / oh)
    else:
        nw = 256
        nh = 256 * (oh / ow)

    p_t = int(32 // 2)
    key_pos = p_t

    anno_entity = video_frame_bbox[sample_id]

    for i, bbox in enumerate(anno_entity["bboxes"]):
        label_tmp = np.zeros((num_classes, ))
        acts_p = anno_entity["acts"][i]
        for l in acts_p:
            label_tmp[l] = 1

        if np.sum(label_tmp) == 0: continue
        p_x = np.int_(bbox[0] * nw)
        p_y = np.int_(bbox[1] * nh)
        p_w = np.int_(bbox[2] * nw)
        p_h = np.int_(bbox[3] * nh)

        boxes.append([p_t, p_x, p_y, p_w, p_h])
        classes.append(label_tmp)

    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
    boxes[:, 1::3].clamp_(min=0, max=int(nw))
    boxes[:, 2::3].clamp_(min=0, max=nh)

    if boxes.shape[0]:
        raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=0)
    else:
        raw_boxes = boxes
    classes = np.array(classes)
    classes = torch.as_tensor(classes, dtype=torch.float32).reshape(-1, num_classes)

    target["image_id"] = [str(sample_id).replace(",", "_"), key_pos]
    target['boxes'] = boxes
    target['raw_boxes'] = raw_boxes
    target["labels"] = classes
    target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
    target["size"] = torch.as_tensor([int(nh), int(nw)])
    # self.index_cnt = self.index_cnt + 1

    return target


def loadvideo(start_img, vid, frame_key):
    frame_path = "/mnt/tmp/frames/{}"
    video_frame_path = frame_path.format(vid)
    video_frame_list = sorted(glob(video_frame_path + '/*.jpg'))

    if len(video_frame_list) == 0:
        print("path doesnt exist", video_frame_path)
        return [], []
    
    target = load_annotation(frame_key, video_frame_list)

    start_img = np.max(start_img, 0)
    end_img = start_img + 32 * 2
    indx_img = list(np.clip(range(start_img, end_img, 2), 0, len(video_frame_list) - 1))
    buffer = []
    for frame_idx in indx_img:
        tmp = Image.open(video_frame_list[frame_idx])
        tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
        buffer.append(tmp)

    return buffer, target


key_frame_candidates = []
GT = {}
for gpu_num in range(8):
    detection = '/mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/Dab_hier_270-39_0430/{}.txt'.format(gpu_num) #numbers are changeable
    gt = '/mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/Dab_hier_270-39_0430/GT_{}.txt'.format(gpu_num)

    with open(gt) as f:
        for line in f.readlines():
            img_id = line.split(' ')[0]
            annotation = [int(float(n)) for n in line.split('[')[1].split(']')[0].split(',')]
            multi_hot_obj_label = annotation[6:]
            key_frame_candidates.append(img_id)
            if not img_id in GT:
                GT[img_id] = []
            GT[img_id].append(multi_hot_obj_label)


class_dict_of_offset_size = {}
sim_thres = 0.4
# ind = random.randint(0, len(key_frame_candidates)-1)
display_freq = 200
sampled_key_frames = random.sample(key_frame_candidates, 5000)

for ind, key_frame in enumerate(sampled_key_frames):
    if ind%display_freq == 0:
        print(ind, len(sampled_key_frames))
    frame_key = key_frame_candidates[ind]
    frame_second = frame_key.split("_")[-1]
    vid = "_".join(frame_key.split('_')[:-1])

    timef = int(frame_second) - 900
    start_img = np.max((timef * 30 - 32 // 2 * 2, 0))

    frame_key = ",".join([vid, frame_second])
    imgs, target = loadvideo(start_img, vid, frame_key)

    """
    start_img: start_img number, int
    vid: xeGWXqSvC-8, CMCPhm2L400, Gcp-cj3bmIY
    frame_key: 0911, 1274, 1725

    """
    orig_vid = imgs
    imgs, target = transforms(imgs, target)
    ho,wo = imgs[0].shape[-2], imgs[0].shape[-1]
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.permute(1, 0, 2, 3)

    # print(len(imgs), imgs[0].shape, target)

    device = "cuda"
    model = model.to(device)
    imgs = imgs.to(device)

    offsets = []
    cls_outputs = []
    input = imgs.unsqueeze(0)

    hooks = [
        # model.backbone.body.layer4[-2].register_forward_hook(
        #     lambda self, input, output: conv_features.append(output)
        # ),
        model.module.class_embed.register_forward_hook(
            lambda self, input, output: cls_outputs.append(output)
        ),
        model.module.transformer.decoder.offset_embed.layers[-1].register_forward_hook(
            lambda self, input, output: offsets.append(output)
        ),
    ]

    outputs = model(imgs.unsqueeze(0))

    for hook in hooks:
        hook.remove()

    offsets = offsets[0][:, 16, :]
    cls_outputs = cls_outputs[0][-1].transpose(0,1).contiguous()[:, 16, :].sigmoid()
    gt_query_sim = torch.matmul(torch.as_tensor(GT[key_frame], device=cls_outputs.device, dtype=torch.double), cls_outputs.T.double()) # N x 15
    query_id = torch.zeros(15, device=gt_query_sim.device, dtype=torch.bool)
    for sim in gt_query_sim:
        query_id = query_id + (sim > sim_thres)
    
    query_id = query_id.nonzero()

    offset_size = offsets[:, 0:2].norm(dim=1).detach().cpu()
    for n in range(15):
        if not n in query_id:
            continue
        cls_to_add = (cls_outputs[n] > 0.7).nonzero()
        for cls in cls_to_add:
            if not cls in class_dict_of_offset_size:
                class_dict_of_offset_size[int(cls)] = []
            class_dict_of_offset_size[int(cls)].append(offset_size[n])

class_dict_of_offset_size_ = {}
for n in range(80):
    try:
        class_dict_of_offset_size_[items[n+1]] = np.mean(class_dict_of_offset_size[n])
    except:
        pass

print({k: v for k, v in sorted(class_dict_of_offset_size_.items(), key=lambda item: item[1])})