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

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, data_file: str = 'ava', binary_loss: bool = False, before: bool = False):
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
        bs, t, num_queries = outputs["pred_logits"].shape[0:3]
        num_classes = outputs["pred_logits"].shape[-1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)          # bs*t, nq, 4
        # Also concat the target labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_bbox = tgt_bbox[:,1:]                               # bs*t, 4
        tgt_ids = torch.cat([v["labels"] for v in targets]) 
        # tgt_ids = F.one_hot(tgt_ids, num_classes=num_classes) # bs*t, 22
        # iou3d = []
        # pad = torch.zeros((len(tgt_bbox), 1), device=tgt_bbox.device)

        # if len(tgt_bbox) != out_bbox.shape[0]: print([len(v["boxes"]) for v in targets], len(targets), out_bbox.size(0))
        
        # _tgt_bbox = torch.cat([pad, box_cxcywh_to_xyxy(tgt_bbox)], -1)
        # _out_bbox = torch.cat([pad[:, None].repeat(1, out_bbox.size(1), 1), box_cxcywh_to_xyxy(out_bbox)], -1)
        # for n in range(out_bbox.size(1)):
            # iou3d.append(compute_video_map.iou3d_voc(_out_bbox[:, n, :].detach().cpu().numpy(), _tgt_bbox.detach().cpu().numpy()))
        # iou3d = torch.tensor(iou3d, device=tgt_bbox.device)[:, None] # n_q, 1 #3d iou

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox.reshape(bs*t, num_queries, 4), tgt_bbox[:, None], p=1).reshape(bs, t, num_queries, 1)
        # bs, t, nq, 1
        
        # Compute the giou cost betwen boxes
        cost_giou = -batched_generalized_box_iou(box_cxcywh_to_xyxy(out_bbox).reshape(-1, num_queries, 4), box_cxcywh_to_xyxy(tgt_bbox).reshape(-1, 1, 4)).reshape(bs, t, num_queries, 1)
        # bs, t, nq, 1

        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # bs*t, nq, num_classes
        

        # print(outputs["pred_logits_b"][..., 1:2].shape, bs, t)
        # cost_class = -out_prob[..., tgt_ids[0]].reshape(bs, t, num_queries, 1)# * outputs["pred_logits_b"].softmax(-1)[..., 1:2]
        # bs, t, nq, 1

        out_prob_b = outputs["pred_logits_b"].softmax(-1)
        cost_class = -out_prob_b[..., 1:2]
        # Final cost matrix
        if not self.binary_loss:
            cost_class_b = 0
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # + cost_class_b
        # bs, t, num_queries, len(tgt_bbox)
        C = C.view(bs*t, num_queries, -1).cpu()
        # import pdb; pdb.set_trace()
        # sizes = [len(v["boxes"]) for v in targets]

        indices = []
        for i in range(bs):
            for j in range(t):
                indices.append(linear_sum_assignment(C[i*t+j]))
        lst = [int(j) for (j,k) in indices]
        idx = max(set(lst), key=lst.count)
        # len(indices): bs * t
        # each element: b번째 batch의 t번째 frame에 해당하는 GT에는 indices[b*T+t] 번째 query가 matching됨
        # element may look like, (array([3]), array([0])): GT box는 하나이기 때문에 두 번째 원소는 항상 array([0])임.

        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # res = []
        # for i, j in indices:
        #     res.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
        #
        # return res
        return [(torch.as_tensor([idx], dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, data_file=cfg.CONFIG.DATA.DATASET_NAME, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE)
