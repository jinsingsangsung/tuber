# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def sigmoid_focal_loss(out_prob, tgt_classes, reduction="none", alpha=0.25, gamma=2.0):
    """
    out_prob: bs*n_q, num_classes (after sigmoid)
    tgt_classes: bs~num_boxes, num_classes
    
    output: bs*n_q, 1
    """
    # tgt_classes = target_classes[:, None, :].repeat(1,N,1).flatten(0,1)
    focal_loss = -torch.matmul((1-out_prob)**gamma*out_prob.log(), tgt_classes.T) + torch.matmul(out_prob**gamma*(1-out_prob).log(), (1-tgt_classes).T)

    # ce_loss = F.binary_cross_entropy_with_logits(out_prob, tgt_classes, reduction="none")
    if alpha >= 0:
        output = alpha * focal_loss
    if reduction == "none":
        pass
    elif reduction == "sum":
        output = output.sum(dim=-1, keepdim=True)
    elif reduction == "mean":
        output = output.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return output


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
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

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
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # bs, num_queries = outputs["pred_boxes"].shape[:2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # Also concat the target labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_bbox = tgt_bbox[:,1:]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # out_classes = outputs["pred_logits"].flatten(0,1).sigmoid()
        # tgt_classes = torch.cat([v["labels"] for v in targets])
        # cost_class = -(torch.mm(out_classes, tgt_classes.T) + torch.mm(1 - out_classes, 1 - tgt_classes.T))/out_classes.shape[-1]

        # cost_class = torch.cdist(out_classes, tgt_classes, p=1)
        # out_prob = outputs["pred_logits_b"].flatten(0, 1).softmax(-1)
        # cost_class = -out_prob[:, 1:2].repeat(1, len(tgt_bbox))

        tgt_classes = torch.cat([v["labels"] for v in targets])
        out_prob = outputs["pred_logits"].flatten(0,1).sigmoid()
        cost_class = sigmoid_focal_loss(out_prob, tgt_classes)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox+ self.cost_giou * cost_giou + self.cost_class * cost_class 
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, data_file=cfg.CONFIG.DATA.DATASET_NAME, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE)
