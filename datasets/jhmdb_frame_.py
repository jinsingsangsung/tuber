"""
The code refers to https://github.com/vkalogeiton/caffe/blob/act-detector/act-detector-scripts/ACT_datalayer.py
Modified by Jiaojiao Zhao
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

class VideoDataset(Dataset):

    def __init__(self, directory, video_path, transforms, clip_len=8, crop_size=224, resize_size=256,
                 mode='train', split=0):
        self.directory = directory
        cache_file = os.path.join(directory, 'JHMDB-GT.pkl')
        assert os.path.isfile(cache_file), "Missing cache file for dataset"

        with open(cache_file, 'rb') as fid:
            dataset = pickle.load(fid, encoding='iso-8859-1')

        self.video_path = video_path
        self._transforms = transforms
        self.dataset = dataset
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.index_cnt = 0
        assert split in [0,1,2]
        # get a list of videos
        if mode == 'val' or mode == 'test':
            self.dataset_samples = self.dataset['test_videos'][split] # split number 0, 1, 2
        elif mode == 'train':
            self.dataset_samples = self.dataset['train_videos'][split]

        self.index_to_sample = [vid for vid in self.dataset_samples]
        max_vid_len = max([self.dataset['nframes'][vid] for vid in self.dataset['nframes'].keys()])
        # total_tubes = sum([len(self.dataset['gttubes'][k].values()) for k in self.dataset['gttubes'].keys()])

        assert max_vid_len <= clip_len,\
            "max video length of the dataset is {}, and clip length (currently {}) needs to be same or larger than that".format(max_vid_len, clip_len)

        print(self.index_to_sample.__len__(), "{} videos indexed".format(mode))

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
        assert target["boxes"].shape[0] == self.clip_len
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)
        assert target["boxes"].shape[0] == self.clip_len
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

        boxes, classes = [], []
        target = {}
        vis = [0]
        tube_len = []

        oh = self.dataset['resolution'][sample_id][0]
        ow = self.dataset['resolution'][sample_id][1]

        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        for ilabel, tubes in self.dataset['gttubes'][sample_id].items():
            # self.max_person = len(tubes) if self.max_person < len(tubes) else self.max_person
            # self.person_size = len(tubes)
            # in JHMDB, there is only one tube per video: thus, len(tubes) = 1
            for t in tubes:
                box_ = t[:, 0:5] # all frames
                tube = []
                # resizing process
                if len(box_[0]) > 0: # if box is valid
                    for box in box_:
                        p_x1 = np.int_(box[1] / ow * nw)
                        p_y1 = np.int_(box[2] / oh * nh)
                        p_x2 = np.int_(box[3] / ow * nw)
                        p_y2 = np.int_(box[4] / oh * nh)
                        tube.append([box[0], p_x1, p_y1, p_x2, p_y2])
                        classes.append(np.clip(ilabel, 0, 21))
                    boxes.append(tube)
                    tube_len.append(len(t))

                    vis[0] = 1

        if self.mode == 'test' and False:
            classes = torch.as_tensor(classes, dtype=torch.int64)
            # print('classes', classes.shape)

            target["vid_id"] = [str(sample_id)]
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            self.index_cnt = self.index_cnt + 1

        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5) # 1, t, 5 -> t, 5
            boxes[:, 1::3].clamp_(min=0, max=nw)
            boxes[:, 2::3].clamp_(min=0, max=nh)

            # make len(boxes) to self.clip_len
            front_pad = (self.clip_len - len(boxes))//2
            end_pad = self.clip_len - len(boxes) - front_pad
            boxes = F.pad(boxes[None, None, ...], (0, 0, front_pad, end_pad), mode="replicate").squeeze()
            assert len(boxes) == self.clip_len

            if boxes.shape[0]: # equals clip_len
                raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt) # put index number in the first column
            else:
                raw_boxes = boxes

            classes = torch.as_tensor(classes, dtype=torch.int64)
            classes = F.pad(classes, (front_pad, end_pad), value=21) # Note that every frame has the same class label
            
            # print('classes', classes.shape)


            
            target["image_id"] = [str(sample_id).replace("/", "_")]
            target['boxes'] = boxes
            target['raw_boxes'] = raw_boxes
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            target["vis"] = torch.as_tensor(vis)
            target['front_pad'] = torch.tensor(front_pad)
            target['end_pad'] = torch.tensor(end_pad)
            target['tube_len'] = torch.tensor(tube_len)
            self.index_cnt = self.index_cnt + 1

        return target 

    # load the video based on keyframe
    def loadvideo(self, sample_id, target):
        from PIL import Image
        import numpy as np
        ## TODO: extend targets to 40 frames when video_len < clip_len
        buffer = []
        # if len(glob(self.video_path + "/" + sample_id + "/*.jpg")) < 66:
        #     print(111)
        # start = max(mid_point - p_t, 0)
        # end = min(mid_point + self.clip_len - p_t, self.dataset["nframes"][sample_id] - 1)
        resolution = self.dataset["resolution"][sample_id] # tuple
        end = self.dataset["nframes"][sample_id] - 1
        frame_ids_ = [s for s in range(end)]
        if len(frame_ids_) < self.clip_len:
            front_size = target["front_pad"]
            front = [0 for _ in range(front_size)]
            back = [end for _ in range(self.clip_len - len(frame_ids_) - front_size)]
            frame_ids_ = front + frame_ids_ + back
        assert len(frame_ids_) == self.clip_len
        for j, frame_idx in enumerate(frame_ids_):
            if j < front_size or j >= front_size + self.dataset["nframes"][sample_id]:
                tmp = Image.new(mode="RGB", size = resolution)
            else:
                tmp = Image.open(os.path.join(self.video_path, sample_id, "{:0>5}.png".format(frame_idx + 1)))
            try:
                tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            except:
                print(target)
                raise "error"
            # buffer.append(np.array(tmp))
            buffer.append(tmp)
        # buffer = np.stack(buffer, axis=0)
        
        # imgs = []
        # for i in range(buffer.shape[0]):
            # imgs.append(Image.fromarray(buffer[i, :, :, :].astype(np.uint8)))
        # return imgs
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
            T.ColorJitter(sat_shift=cfg.CONFIG.AUG.COLOR_JITTER,val_shift=cfg.CONFIG.AUG.COLOR_JITTER,),
            T.PCAJitter(alphastd=0.1,
                        eigval=np.array(cfg.CONFIG.AUG.TRAIN_PCA_EIGVAL).astype(np.float32),
                        eigvec=np.array(cfg.CONFIG.AUG.TRAIN_PCA_EIGVEC).astype(np.float32),),
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
                                 mode="train",
                                 split=cfg.CONFIG.DATA.SPLIT,)

    val_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                               video_path=cfg.CONFIG.DATA.DATA_PATH,
                               transforms=make_transforms("val", cfg),
                               clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                               resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                               crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                               mode="val",
                               split=cfg.CONFIG.DATA.SPLIT,)

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
