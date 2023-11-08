"""
changes tuber result file (of ViT-B to AVA format of .txt file)
"""

import torch
from models.transformer.util import box_ops
from datasets.ava_frame import build_dataloader
from pipelines.video_action_recognition_config import get_cfg_defaults
import json

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from models.transformer.util import box_ops
from datasets.ava_frame import build_dataloader
from pipelines.video_action_recognition_config import get_cfg_defaults
from tqdm import tqdm
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh
from utils.misc import inverse_sigmoid



def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

input = "full_jinsung"
# input = "full_tuber"
# tuber_txt = open(f"/mnt/video_nfs4/users/jinsung/results/tubelet-transformer/{input}.txt").readlines()
# gt_txt = open(f"/mnt/video_nfs4/users/jinsung/results/tubelet-transformer/full_GT.txt").readlines()
tuber_txt_ = "../{input}.txt"
gt_txt_ = "../GT_{input}.txt"
"""
sample tuber csv:
vid_id,frame_id, x1, y1, x2, y2, cls_id, confidence_score
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,1,0.0000
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,2,0.0000
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,3,0.0000
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,4,0.0386
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,5,0.0000
1j20qq1JyX4,0902,0.022,0.113,0.493,0.987,6,0.0000

output format:
Gvp-cj3bmIY_1156 [x1, y1, x2, y2, class_i, ...]

"""

# cfg = get_cfg_defaults()
# cfg.merge_from_file("./configuration/dumm_config.yaml")
# train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)


# exclude_keys = []
# ff = open("/home/nsml/assets/ava_val_excluded_timestamps_v2.1.csv")
# while True:
#     line = ff.readline().strip()
#     if not line: break
#     exclude_keys.append(line.replace(",", "_"))
# ff.close()

tuber = {}
for j in range(8):
    tuber_txt = open(tuber_txt_.format(j)).readlines()
    for i, line in enumerate(tqdm(tuber_txt)):
        vid_fid = line.split(' [')[0]
        data = line.split(' [')[1].split(']')[0].split(',')
        data = [float(x) for x in data]
        scores = torch.tensor(np.array(data[4:80 + 4]))
        box = torch.tensor(np.array(data[:4]))
        person_score = torch.tensor(np.array(data[-1]))
        if person_score < 0.6:
            continue
        if not vid_fid in tuber.keys():
            tuber[vid_fid] = {"bbox": [], "cls_score": []}
        tuber[vid_fid]["bbox"].append(box)
        tuber[vid_fid]["cls_score"].append(scores)
    
print(f"creating {input} prediction dict is done")


tuber_gt = {}
for k in range(8):
    gt_txt = open(gt_txt_.format(k)).readlines()
    for i, line in enumerate(tqdm(gt_txt)):
        vid_fid = line.split(' [')[0]
        data = line.split(' [')[1].split(']')[0].split(',')
        data = [float(x) for x in data]
        scores = torch.tensor(np.array(data[6:80 + 6]))
        box = torch.tensor(np.array(data[2:6]))
        if not vid_fid in tuber_gt.keys():
            tuber_gt[vid_fid] = {"bbox": [], "cls_score": []}
        tuber_gt[vid_fid]["bbox"].append(box)
        tuber_gt[vid_fid]["cls_score"].append(scores)

print(f"creating {input} annotation dict is done")

whole_detections = {}

detection = {}
model_name = input
threshold = 0.6

accuracy_bin_p = torch.zeros(1,80)
accuracy_bin_cnt = torch.zeros(1,80)
TP = torch.zeros(1,80) # 있는데 있다고 함
FP = torch.zeros(1,80) # 없는데 있다고 함
FN = torch.zeros(1,80) # 있는데 없다고 함
TN = torch.zeros(1,80) # 없는데 없다고 함

box_cnt = 0
omitted_frames = []

for vid_fid in tqdm(list(tuber_gt.keys())):
    try:
        out_cls = torch.stack(tuber[vid_fid]["cls_score"], dim=0) # num_actors, 80
        tgt_cls = torch.stack(tuber_gt[vid_fid]["cls_score"], dim=0) # num_actors, 80
    except:
        omitted_frames.append(vid_fid)
        tgt_cls = torch.stack(tuber_gt[vid_fid]["cls_score"], dim=0) # num_actors, 80
        out_cls = torch.zeros_like(tgt_cls)        
        continue

    out_cls_set = out_cls.sum(dim=0)
    tgt_cls_set = tgt_cls.sum(dim=0)
    
    out_cls_set[out_cls_set>=threshold] = 1
    out_cls_set[out_cls_set<threshold] = 0
    
    tgt_cls_set = tgt_cls_set.clamp(max=1)
    FN += (tgt_cls_set > out_cls_set).float()
    TP += (tgt_cls_set==1).float() * (out_cls_set==1).float()
    FP += (tgt_cls_set < out_cls_set).float()
    TN += (tgt_cls_set==0).float() * (out_cls_set==0).float()


classwise_acc = ((TP + TN) / (TP + TN + FP + FN)).nan_to_num()
classwise_precision = (TP / (TP+FP)).nan_to_num()
classwise_recall = (TP / (TP+FN)).nan_to_num()
# _val_loss = val_loss / box_cnt

print(f"============ result for {model_name}: ")    
# print("val loss: ", _val_loss.float().item())
print("box_cnt: ", box_cnt)
print("frames used: ", len(tuber_gt.keys()) - len(omitted_frames))
print("classification accuracy: ", classwise_acc.mean().float().item())
precision = classwise_precision.mean().float().item()
recall = classwise_recall.mean().float().item()
print("classification precision: ", precision)
print("classification recall: ", recall)
print("F1 score:", 2* (precision*recall) / (precision + recall))
# print("class-wise accuracy/precision/recall: ")
# for i, (acc, prec, rec) in enumerate(zip(classwise_acc[0], classwise_precision[0], classwise_recall[0])):
#     print(f"{items[i+1]} acc: {acc}")
#     print(f"{items[i+1]} precision: {prec}")
#     print(f"{items[i+1]} recall: {rec}")
# print("Localization avg G-IoU: ", _giou.float().item())

