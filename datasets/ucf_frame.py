"""
The code refers to https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT_datalayer.py
Modified by Jiaojiao Zhao
Modified by Jinsung Lee
"""

import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils.data
import torch.nn.functional as F
import datasets.video_transforms as T
from utils.misc import collate_fn
from glob import glob
import random

# Assisting function for finding a good/bad tubelet
# def tubelet_in_tube(tube, i, K):
#     # True if all frames from i to (i + K - 1) are inside tube
#     # it's sufficient to just check the first and last frame.
#     return (i in tube[:, 0] and i + K - 1 in tube[:, 0])

def tubelet_in_tube(tube, i, K):
    # True if all frames from i to (i + K - 1) are inside tube
    # it's sufficient to just check the first and last frame.
    return i in tube[:, 0]


def tubelet_out_tube(tube, i, K):
    # True if all frames between i and (i + K - 1) are outside of tube
    return all([j not in tube[:, 0] for j in range(i, i + K)])


def tubelet_in_out_tubes(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # all frames from i to (i + K - 1) are either inside (tubelet_in_tube)
    # or outside (tubelet_out_tube) the tubes.
    return all([tubelet_in_tube(tube, i, K) or tubelet_out_tube(tube, i, K) for tube in tube_list])


def tubelet_has_gt(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # the tubelet starting spanning from [i to (i + K - 1)]
    # is inside (tubelet_in_tube) at least a tube in tube_list.
    return any([tubelet_in_tube(tube, i, K) for tube in tube_list])


CLASSES = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
        'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
        'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

class VideoDataset(Dataset):

    def __init__(self, directory, video_path, transforms, clip_len=32, crop_size=224, resize_size=256,
                 mode='train'):
        self.directory = directory
        cache_file = os.path.join(directory, 'UCF101v2-GT.pkl')
        assert os.path.isfile(cache_file), "Missing cache file for dataset"

        with open(cache_file, 'rb') as fid:
            dataset = pickle.load(fid, encoding='iso-8859-1')

        self.video_path = video_path
        self._transforms = transforms
        self.dataset = dataset
        self.mode = mode
        self.clip_len = clip_len
        assert clip_len%2 == 0
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.index_cnt = 0
        self.num_classes = len(CLASSES)
        # get a list of videos
        if mode == 'val' or mode == 'test':
            self.dataset_samples = self.dataset['test_videos'][0] # split number 0, 1, 2
        elif mode == 'train':
            self.dataset_samples = self.dataset['train_videos'][0]
        self.video_to_sample = [vid for vid in self.dataset_samples]
        # max_vid_len = max([self.dataset['nframes'][vid] for vid in self.dataset['nframes'].keys()])
        min_vid_len = min([self.dataset['nframes'][vid] for vid in self.dataset['nframes'].keys()])
        # print("max:", max_vid_len, "min:", min_vid_len): 900, 32
        # total_tubes = sum([len(self.dataset['gttubes'][k].values()) for k in self.dataset['gttubes'].keys()])

        assert min_vid_len >= clip_len,\
            "min video length of the dataset is {}, and clip length (currently {}) needs to be same or larger than that".format(min_vid_len, clip_len)

        print(self.video_to_sample.__len__(), "{} videos indexed".format(mode))
        index_to_sample = []
        total_clips = 0
        for vid in self.video_to_sample:
            nframes = self.dataset['nframes'][vid] # note that not all frames are properly annotated
            # nframes = len(list(self.dataset['gttubes'][vid].values())[0][0]) # number of annotated frames
            num_clips = nframes // clip_len + int(nframes%clip_len != 0)
            amount_to_pad = clip_len-nframes%clip_len
            front_pad = amount_to_pad // 2
            end_pad = amount_to_pad - front_pad
            # let's take the center frame of the clip
            index_to_sample.extend([(vid, i*clip_len+(clip_len//2)-front_pad+1, front_pad, end_pad, nframes) for i in range(num_clips)])
            total_clips += num_clips
        print("total {} clips: {}".format(mode, total_clips))
        self.index_to_sample = index_to_sample
        self.labelmap = self.dataset['labels']
        self.max_person = 0
        self.person_size = 0

    # def check_video(self, vid):
    #     frames_ = glob(self.directory + "/rgb-images/" + vid + "/*.jpg")
    #     if len(frames_) < 66:
    #         print(vid, len(frames_))

    def __getitem__(self, index):

        sample_id = self.index_to_sample[index]

        target = self.load_annotation(sample_id)
        imgs = self.loadvideo(sample_id, target)
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)
        if self.mode == 'test':
            if target['boxes'].shape[0] == 0:
                target['boxes'] = torch.concat([target["boxes"], torch.from_numpy(np.array([[0, 0, 0, 1, 1]]))])
                target['labels'] = torch.concat([target["labels"], torch.from_numpy(np.array([0]))])
                target['area'] = torch.concat([target["area"], torch.from_numpy(np.array([30]))])
                target['raw_boxes'] = torch.concat([target["raw_boxes"], torch.from_numpy(np.array([[0, 0, 0, 0, 1, 1]]))])

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)
        return imgs, target

    def load_annotation(self, sample_id):

        # print('sample_id',sample_id)
        vid_id, c_frame, front_pad, end_pad, nframes = sample_id
        boxes, classes = [], []
        target = {}
        vis = [0]
        tube_len = []

        oh = self.dataset['resolution'][vid_id][0]
        ow = self.dataset['resolution'][vid_id][1]

        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        for ilabel, tubes in self.dataset['gttubes'][vid_id].items():
            # self.max_person = len(tubes) if self.max_person < len(tubes) else self.max_person
            # self.person_size = len(tubes)
            # in UCF, there can be several tube per video: thus, len(tubes) != 1

            # see if there is a overlapping region between clip and GT
            clip_start_frame = c_frame-self.clip_len//2
            clip_end_frame = c_frame+self.clip_len//2-1
            pad_front = True if clip_start_frame <= 0 else False
            pad_end = True if clip_end_frame > nframes else False
            # if len(tubes) > 1:
            #     print(vid_id, len(tubes))
            for t in tubes:
                box_ = t[:, 0:5] # all frames
                tube = []
                gt_start_frame = int(box_[0][0])
                gt_end_frame = int(box_[-1][0])

                # case 1: GT and clip do not overlap
                if clip_end_frame < gt_start_frame or clip_start_frame > gt_end_frame:
                    classes_ = [self.num_classes for _ in range(self.clip_len)]
                    tube.extend([[n, -1, -1, -1, -1] for n in range(clip_start_frame, clip_end_frame+1)])
                    boxes.append(tube)
                    tube_len.append(self.clip_len)
                    vis[0] = 0
                    classes.append(classes_)
                
                # case2: clip overlap the front part of the GT
                elif clip_end_frame >= gt_start_frame and clip_start_frame <= gt_start_frame and clip_end_frame < gt_end_frame:
                    classes_ = [self.num_classes for _ in range(gt_start_frame - clip_start_frame)]
                    tube.extend([[n, -1, -1, -1, -1] for n in range(clip_start_frame, gt_start_frame)])
                    # resizing process
                    if len(box_[0]) > 0: # if box is valid
                        for box in box_[:-gt_end_frame+clip_end_frame,:]:
                            p_x1 = np.int_(box[1] / ow * nw)
                            p_y1 = np.int_(box[2] / oh * nh)
                            p_x2 = np.int_(box[3] / ow * nw)
                            p_y2 = np.int_(box[4] / oh * nh)
                            tube.append([box[0], p_x1, p_y1, p_x2, p_y2])
                            classes_.append(np.clip(ilabel, 0, 24))
                    boxes.append(tube)
                    classes.append(classes_)
                    tube_len.append(len(box_[:-gt_end_frame+clip_end_frame,:]))
                    vis[0] = 1        
                    if len(tube) != self.clip_len:
                        print("case2:", clip_start_frame, clip_end_frame, gt_start_frame, gt_end_frame, len(tube))            

                # case3: clip overlap the end part of the GT
                elif clip_end_frame >= gt_end_frame and clip_start_frame >= gt_start_frame:
                    classes_ = []
                    if len(box_[0]) > 0: # if box is valid
                        for box in box_[clip_start_frame-gt_start_frame:,:]:
                            p_x1 = np.int_(box[1] / ow * nw)
                            p_y1 = np.int_(box[2] / oh * nh)
                            p_x2 = np.int_(box[3] / ow * nw)
                            p_y2 = np.int_(box[4] / oh * nh)
                            tube.append([box[0], p_x1, p_y1, p_x2, p_y2])
                            classes_.append(np.clip(ilabel, 0, 24))
                    tube.extend([[n, -1, -1, -1, -1] for n in range(gt_end_frame+1, clip_end_frame+1)])
                    classes_.extend([self.num_classes for _ in range(clip_end_frame-gt_end_frame)])
                    boxes.append(tube)
                    classes.append(classes_)
                    tube_len.append(len(box_[clip_start_frame-gt_start_frame:,:]))
                    if len(tube) != self.clip_len:
                        print("case3:", clip_start_frame, clip_end_frame, gt_start_frame, gt_end_frame, len(tube))                    
                    vis[0] = 1

                # case4: clip overlaps inside the GT
                elif clip_start_frame > gt_start_frame and clip_end_frame < gt_end_frame:
                    classes_ = []
                    if len(box_[0]) > 0: # if box is valid
                        for box in box_[clip_start_frame-gt_start_frame: clip_end_frame-gt_end_frame,:]:
                            p_x1 = np.int_(box[1] / ow * nw)
                            p_y1 = np.int_(box[2] / oh * nh)
                            p_x2 = np.int_(box[3] / ow * nw)
                            p_y2 = np.int_(box[4] / oh * nh)
                            tube.append([box[0], p_x1, p_y1, p_x2, p_y2])
                            classes_.append(np.clip(ilabel, 0, 24))
                    boxes.append(tube)
                    tube_len.append(self.clip_len)   
                    classes.append(classes_)

                    vis[0] = 1 
                    if len(tube) != self.clip_len:
                        print("case4:", clip_start_frame, clip_end_frame, gt_start_frame, gt_end_frame, len(tube))
                
                # case5: clip overlaps over the whole GT
                elif clip_start_frame <= gt_start_frame and clip_end_frame >= gt_end_frame:
                    classes_ = [self.num_classes for _ in range(gt_start_frame - clip_start_frame)]
                    tube.extend([[n, -1, -1, -1, -1] for n in range(clip_start_frame, gt_start_frame)])
                    # resizing process
                    if len(box_[0]) > 0: # if box is valid
                        for box in box_:
                            p_x1 = np.int_(box[1] / ow * nw)
                            p_y1 = np.int_(box[2] / oh * nh)
                            p_x2 = np.int_(box[3] / ow * nw)
                            p_y2 = np.int_(box[4] / oh * nh)
                            tube.append([box[0], p_x1, p_y1, p_x2, p_y2])
                            classes_.append(np.clip(ilabel, 0, 24))
                    classes_.extend([self.num_classes for _ in range(clip_end_frame - gt_end_frame)])
                    tube.extend([[n, -1, -1, -1, -1] for n in range(gt_end_frame, clip_end_frame)])
                    boxes.append(tube)
                    classes.append(classes_)
                    tube_len.append(self.clip_len)   
                    if len(tube) != self.clip_len:
                        print("case5:", clip_start_frame, clip_end_frame, gt_start_frame, gt_end_frame, len(tube))                    


                else:
                    print("!edge case detected!")
                    print(clip_start_frame, clip_end_frame, gt_start_frame, gt_end_frame)
                    raise AssertionError

        if self.mode == 'test' and False:
            classes = torch.as_tensor(classes, dtype=torch.int64)
            # print('classes', classes.shape)

            target["vid_id"] = [str(sample_id)]
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            self.index_cnt = self.index_cnt + 1

        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).flatten(0,1) # num_tubes*clip_len, 5
            boxes[..., 1::3].clamp_(min=-1, max=nw)
            boxes[..., 2::3].clamp_(min=-1, max=nh)
        
            # assert len(classes[0]) == self.clip_len
            if boxes.shape[0]: # equals num_tubes*clip_len
                raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt) # put index number in the first column
            else:
                raw_boxes = boxes

            classes = torch.as_tensor(classes, dtype=torch.int64)
            # print(classes.shape) num_boxes, clip_len
                
            target["image_id"] = [str(vid_id).replace("/", "_")]
            target['boxes'] = boxes
            target['raw_boxes'] = raw_boxes
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            target["vis"] = torch.as_tensor(vis)
            target['front_pad'] = torch.tensor(front_pad) if pad_front else torch.tensor(0)
            target['end_pad'] = torch.tensor(end_pad) if pad_end else torch.tensor(0)
            target['tube_len'] = torch.tensor(tube_len)
            self.index_cnt = self.index_cnt + 1

        return target 

    # load the video based on keyframe
    def loadvideo(self, sample_id, target):
        from PIL import Image
        import numpy as np
        buffer = []
        vid_id, c_frame, front_pad, end_pad, nframes = sample_id
        clip_start_frame = c_frame-self.clip_len//2
        clip_end_frame = c_frame+self.clip_len//2-1
        if clip_start_frame <= 0:
            frame_ids_ = [1 for _ in range(front_pad)]
            frame_ids_.extend([s+1 for s in range(clip_end_frame)])
        elif clip_end_frame > nframes:
            frame_ids_ = [s for s in range(clip_start_frame, nframes+1)]
            frame_ids_.extend([nframes for _ in range(end_pad)])
        else:
            frame_ids_ = [s for s in range(clip_start_frame, clip_end_frame+1)]

        assert len(frame_ids_) == self.clip_len
        
        for frame_idx in frame_ids_:
            tmp = Image.open(os.path.join(self.video_path, vid_id, "{:0>5}.jpg".format(frame_idx)))
            try:
                tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            except:
                print(target)
                raise "error"
            # buffer.append(np.array(tmp))
            buffer.append(tmp)
        # buffer = np.stack(buffer, axis=0)

        return buffer

    def __len__(self):
        return len(self.index_to_sample)


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
            # T.HorizontalFlip(),
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])

    if image_set == 'visual':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def build_dataloader(cfg):
    train_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                                 video_path=cfg.CONFIG.DATA.DATA_PATH,
                                 transforms=make_transforms("train", cfg),
                                 clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                                 resize_size=cfg.CONFIG.DATA.IMG_RESHAPE_SIZE,
                                 crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                                 mode="train")

    val_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                               video_path=cfg.CONFIG.DATA.DATA_PATH,
                               transforms=make_transforms("val", cfg),
                               clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                               resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                               crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                               mode="val")

    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, cfg.CONFIG.TRAIN.BATCH_SIZE, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        batch_sampler_train = None

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=(train_sampler is None),
                                               num_workers=9, pin_memory=True, batch_sampler=batch_sampler_train,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True, collate_fn=collate_fn)

    print(cfg.CONFIG.DATA.ANNO_PATH.format("train"), cfg.CONFIG.DATA.ANNO_PATH.format("val"))

    return train_loader, val_loader, train_sampler, val_sampler, None
