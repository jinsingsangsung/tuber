# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, batched_generalized_box_iou
from evaluates.utils import compute_video_map

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, data_file: str = 'ava', binary_loss: bool = False, before: bool = False, clip_len: int = 32):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.data_file = data_file
        self.binary_loss = binary_loss
        self.before = before
        self.clip_len = clip_len
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, t, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, t, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        l = self.clip_len
        bs, t, num_queries, num_classes = outputs["pred_logits"].shape
        num_classes = outputs["pred_logits"].shape[-1]
        out_bbox = outputs["pred_boxes"].permute(1,0,2,3).flatten(1, 2)    # t, bs*nq, 4
        # out_prob = outputs["pred_logits"].permute(1,0,2,3).flatten(1, 2).softmax(-1) # t, bs*nq, num_classes
        # Also concat the target labels
        # tgt_bbox = torch.cat([v["boxes"] for v in targets]) 
        sizes = []
        tgt_bbox = []
        # if torch.cat([v["boxes"] for v in targets]).__len__() > l:
        #     import pdb; pdb.set_trace()
        for v in targets:
            tgt_bbox_ = v["boxes"]
            num_tubes = len(tgt_bbox_) // l
            sizes.append(num_tubes)
            # tgt_bboxes = tgt_bbox_.split(l)
            # tgt_bbox.append(torch.stack([tgt_bboxes[i] for i in range(num_tubes)], dim=0)) # num_tubes x clip_len x 5
            tgt_bbox.append(tgt_bbox_.reshape(num_tubes, -1 ,5))
        tgt_bbox = torch.cat(tgt_bbox, dim=0)[..., 1:]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        invalid_ids = tgt_ids >= num_classes
        assert len(tgt_bbox) == len(tgt_ids)
        # TODO: Matching strategy: tube? or frame? let's go with tube
        # 1. make it generalizable to multiple actors, make a good output of matching indices (take the idea from ava)
        # 2. discard the outputs with non-meaningful labels (actually null tube) how to return the idx?

        tgt_bbox = tgt_bbox.permute(1,0,2).contiguous()
        tgt_ids = tgt_ids.permute(1,0).contiguous()
        invalid_ids = invalid_ids.permute(1,0).contiguous() # clip_len x bs~#
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(tgt_bbox, out_bbox) # clip_len x bs~# x bs*n_q
        cost_bbox[invalid_ids] = torch.zeros(out_bbox.shape[1], device=cost_bbox.device)
        cost_bbox = cost_bbox.permute(0,2,1).contiguous()

        # Compute the giou cost betwen boxes
        cost_giou = -batched_generalized_box_iou(box_cxcywh_to_xyxy(tgt_bbox), box_cxcywh_to_xyxy(out_bbox))
        cost_giou[invalid_ids] = torch.zeros(out_bbox.shape[1], device=cost_bbox.device)
        cost_giou = cost_giou.permute(0,2,1).contiguous()
        out_prob = outputs["pred_logits_b"].permute(1,0,2,3).flatten(1, 2).softmax(-1)
        cost_class = -out_prob[..., 1:2].repeat(1,1,sum(sizes))
        # t, bs*n_q, bs~#

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class
        # t, bs*n_q, bs~#

        # bs, t, num_queries, len(tgt_bbox)
        C = C.view(t, bs, num_queries, -1).cpu()

        indices = []
        output = []
        
        # matched_queries = []
        for j in range(t):
            idx = [linear_sum_assignment(c[i]) for i, c in enumerate(C[j].split(sizes, -1))]

            for batch_id, (queries, keys) in enumerate(idx):
                if len(output) < len(idx):
                    interm = {}
                    for q, k in zip(queries, keys):
                        if interm.get(k) is None:
                            interm[k] = []
                        interm[k].append(q)
                    output.append(interm) 
                else:
                    for q, k in zip(queries, keys):
                        output[batch_id][k].append(q)
            indices.append(idx) # idx length: batch
        output_real = []
        for batch_id, out_dict in enumerate(output):
            interm = []
            for k in out_dict:
                interm.append(torch.tensor([max(set(out_dict[k]), key=out_dict[k].count), k]))
            interm = torch.stack(interm).transpose(0, 1).contiguous()
            output_real.append(interm)
                

        # if (torch.tensor(sizes) > 1).any():
        #     print(output_real)

        # t, num_
        # element may look like, (array([3, 5]), array([1, 0])) # 1번 target은 3, 0번 target은 5번 query matched

        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # res = []
        # for i, j in indices:
        #     res.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
        #
        # return res
        # return [(torch.as_tensor([idx], dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return output_real

def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, data_file=cfg.CONFIG.DATA.DATASET_NAME, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE, clip_len=cfg.CONFIG.DATA.TEMP_LEN)
