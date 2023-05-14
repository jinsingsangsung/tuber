import json

import torch

from utils.utils import read_labelmap
from evaluates.utils import object_detection_evaluation, standard_fields, video_map
import numpy as np
import time
from utils.box_ops import box_iou
import torch
import math


def parse_id():
    activity_list = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
    categories = []
    for i, act_name in enumerate(activity_list):
        categories.append({'id': i + 1, 'name': act_name})
    return categories


class STDetectionEvaluaterUCF(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, tiou_thresholds=[0.5], load_from_dataset=False, class_num=24):
        categories = parse_id()
        self.class_num = class_num
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.video_map_evaluator = []
        self.load_from_dataset = load_from_dataset
        self.exclude_key = []
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories, matching_iou_threshold=iou)
                )
            self.video_map_evaluator.append(
                video_map.VideoMAPEvaluator(categories, matching_iou_threshold=iou)
                )
    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_GT_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}
        all_annots = []
        frame_counter = {}
        for path in file_lst:
            data_ = open(path).readlines()
            for i, line in enumerate(data_):
                image_key = line.split(' [')[0]
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]
                scores = np.array(data[6:])
                if not image_key in frame_counter:
                    frame_counter[image_key] = 0
                if frame_counter[image_key] == 1:
                    continue

                if i < len(data_)-1:
                    if image_key != data_[i+1].split(' [')[0]:
                        frame_counter[image_key] = 1
                else:
                    frame_counter[image_key] = 1

                # if (data[4] - data[2]) * (data[5] - data[3]) < 10:
                #     self.exclude_key.append(image_key)
                #     continue

                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                # scores = np.max(scores, axis=-1, keepdims=True)

                for x in range(len(scores)):
                    if scores[x] <= 1e-2: continue
                    sample_dict_per_image[image_key]['bbox'].append(
                        np.asarray([data[2], data[3], data[4], data[5]], dtype=float)
                    )
                    sample_dict_per_image[image_key]['labels'].append(x + 1)
                    sample_dict_per_image[image_key]['scores'].append(scores[x])
                all_annots.append(line)

        gt_videos = {} # for video-map
        annot_memory = []
        all_annots.sort(key=lambda x: x.split(" [")[0]) # merge GT clips into original videos, remove duplicated lines
        for i, line in enumerate(all_annots):
            image_key = line.split(' [')[0]
            vname = "_".join(image_key.split('_')[:-1])
            data = line.split(' [')[1].split(']')[0].split(',')
            data = [float(x) for x in data]
            scores = np.array(data[6:])
            if not vname in gt_videos:
                gt_videos[vname] = {
                    "tubes": [],
                    "gt_classes": 25
                }
            # gt_videos[vname]["tubes"].append([data[1], data[2], data[3], data[4], data[5]])

            if gt_videos[vname]["gt_classes"] == 25:
                gt_videos[vname]["gt_classes"] = min(int(scores.nonzero()[0])+1, 25)
            
            if i < len(all_annots)-1:
                next_video = "_".join(all_annots[i+1].split(' [')[0].split('_')[:-1])
                annot_memory.append(data)
                if next_video != vname:
                    nframes = len(set([d[1] for d in annot_memory]))
                    ntubes = len(annot_memory) // nframes
                    gt_videos[vname]["tubes"] = [np.array(annot_memory[n::ntubes])[np.array(annot_memory[n::ntubes])[:,-1] != 1][:,1:6] for n in range(ntubes) if (1-np.array(annot_memory[n::ntubes])[:, -1]).any()]
                    # np.asarray(gt_videos[vname]["tubes"]).reshape((ntubes, -1, 5))
                    annot_memory = []
            else:
                annot_memory.append(data)
                nframes = len(set([d[1] for d in annot_memory]))
                ntubes = len(annot_memory) // nframes
                gt_videos[vname]["tubes"] = [np.array(annot_memory[n::ntubes])[np.array(annot_memory[n::ntubes])[:,-1] != 1][:,1:6] for n in range(ntubes) if (1-np.array(annot_memory[n::ntubes])[:, -1]).any()]

        # write into evaluator
        for image_key, info in sample_dict_per_image.items():
            if len(info['bbox']) == 0: continue
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(info['bbox']),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(info['labels'], dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(info['bbox']), dtype=bool)
                    })
            for v_evaluator in self.video_map_evaluator:
                v_evaluator.add_gt(gt_videos)

        print("STDetectionEvaluater: test GT loaded in {:.3f}s".format(time.time() - t_end))

    def load_detection_from_path(self, file_lst):
        # loading data from files
        num_queries = 10
        t_end = time.time()
        sample_dict_per_image = {}
        all_boxes = {} # for video-map
        n = 0
        image_key_dict = {}        
        for path in file_lst:
            print("loading ", path)
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                # if image_key in self.exclude_key:
                #     continue
                if not image_key in image_key_dict:
                    image_key_dict[image_key] = 0
                image_key_dict[image_key] += 1
                if image_key_dict[image_key] > num_queries:
                    continue
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]

                scores = np.array(data[4:self.class_num + 4])

                x = np.argmax(scores)

                # all_boxes[image_key][x+1] = np.asarray([data[0], data[1], data[2], data[3], scores[x]], dtype=float)
                if not image_key in all_boxes:
                    all_boxes[image_key] = {}

                for s in range(len(scores)):
                    if not (s+1) in all_boxes[image_key]:
                        all_boxes[image_key][s+1] = []

                    # if s != x:
                        # all_boxes[image_key][s+1].append([data[0], data[1], data[2], data[3], 0])
                    if s == x:
                        all_boxes[image_key][s+1].append([data[0], data[1], data[2], data[3], scores[x]])
                    
                    # if len(all_boxes[image_key][s+1]) == num_queries:
                    #     all_boxes[image_key][s+1] = np.asarray(all_boxes[image_key][s+1], dtype=float)

                if np.argmax(np.array(data[4:])) == len(np.array(data[4:])) - 1:
                    continue

                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }

                sample_dict_per_image[image_key]['bbox'].append(
                    np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
                )
                sample_dict_per_image[image_key]['labels'].append(x+1)
                sample_dict_per_image[image_key]['scores'].append(scores[x])

                # for x in range(len(scores)):
                #     sample_dict_per_image[image_key]['bbox'].append(
                #         np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
                #     )
                #     sample_dict_per_image[image_key]['labels'].append(x+1)
                #     sample_dict_per_image[image_key]['scores'].append(scores[x])

        for k in list(all_boxes.keys()):
            for s in range(len(scores)):
                all_boxes[k][s+1] = np.asarray(all_boxes[k][s+1], dtype=float)

        print("start adding into evaluator")
        for v_evaluator in self.video_map_evaluator:
            v_evaluator.add_pred(all_boxes)

        count = 0
        for image_key, info in sample_dict_per_image.items():
            if count % 10000 == 0:
                print(count, len(sample_dict_per_image.keys()))
            if len(info['bbox']) == 0:
                print(count)
                continue
            #sorted by confidence:
            boxes, labels, scores = np.vstack(info['bbox']), np.array(info['labels'], dtype=int), np.array(info['scores'], dtype=float)
            index = np.argsort(-scores)
            #print('scores',scores[index])
            #exit()
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes:
                            boxes[index],
                        standard_fields.DetectionResultFields.detection_classes:
                            labels[index],
                        standard_fields.DetectionResultFields.detection_scores:
                            scores[index]
                    })
            count += 1


    def evaluate(self):
        result = {}
        v_result = {}
        mAP = []
        v_mAP = []
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            v_evaluator = self.video_map_evaluator[x]
            metrics = evaluator.evaluate()
            v_metrics = v_evaluator.evaluate_videoAP()
            result.update(metrics)
            v_result.update(v_metrics)
            mAP.append(metrics['PascalBoxes_Precision/mAP@{}IOU'.format(iou)])
            v_mAP.append(v_metrics["video-mAP@{}IOU".format(iou)])
        return mAP, result, v_mAP, v_result

#
# if __name__ == '__main__':
#     evaluater = STDetectionEvaluaterUCF(class_num=24)
#     file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in
#                      range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
#     evaluater.load_GT_from_path(file_path_lst)
#     file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in
#                      range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
#     evaluater.load_detection_from_path(file_path_lst)
#     mAP, metrics = evaluater.evaluate()
#     print(metrics)
