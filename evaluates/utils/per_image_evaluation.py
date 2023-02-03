# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluate Object Detection result on a single image.

Annotate each detected result as true positives or false positive according to
a predefined IOU ratio. Non Maximum Supression is used by default. Multi class
detection is supported by default.
Based on the settings, per image evaluation is either performed on boxes or
on object masks.
"""
import numpy as np

import evaluates.utils.np_box_list as np_box_list
import evaluates.utils.np_box_list_ops as np_box_list_ops

import copy

def detected_class_corrector(
    detected_boxes,
    detected_class_labels,
    detected_scores,
    groundtruth_boxes,
    groundtruth_class_labels,
    groundtruth_is_group_of_list): # copied from _get_overlaps_and_scores_box_mode()

  detected_boxlist = np_box_list.BoxList(detected_boxes)
  gt_non_group_of_boxlist = np_box_list.BoxList(
      groundtruth_boxes[~groundtruth_is_group_of_list])
  iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
  if iou.shape[1] > 0:
    max_overlap_gt_ids = np.argmax(iou, axis=1) # which gt box does the detected box indicate?
    # new_labels = copy.deepcopy(detected_class_labels)
    new_scores = copy.deepcopy(detected_scores)
    for i, label in enumerate(detected_class_labels):
      if iou[i, max_overlap_gt_ids[i]] >= 0.8 and groundtruth_class_labels[max_overlap_gt_ids[i]] == label:
        # new_labels[i] = groundtruth_class_labels[max_overlap_gt_ids[i]]
        new_scores[i] = 1.0
    corrected_detected_scores = np.array(new_scores)
    # print("iou shape: ", iou.shape)
    # print("max_overlap_gt_ids shape: ", max_overlap_gt_ids.shape)
    # how_many_are_corrected = (detected_class_labels == corrected_detected_class_labels)
    # print("amount of correction: ", len(how_many_are_corrected) - sum(how_many_are_corrected), '/', len(how_many_are_corrected))
    # print("detected_class_labels: ", detected_class_labels)
    # print("corrected_detected_class_labels: ", corrected_detected_class_labels)

  else:
    # corrected_detected_scores = detected_scores
    corrected_detected_scores = np.zeros_like(detected_scores)

  return corrected_detected_scores

class PerImageEvaluation(object):
  """Evaluate detection result of a single image."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5):
    """Initialized PerImageEvaluation by evaluation parameters.

    Args:
      num_groundtruth_classes: Number of ground truth object classes
      matching_iou_threshold: A ratio of area intersection to union, which is
          the threshold to consider whether a detection is true positive or not
    """
    self.matching_iou_threshold = matching_iou_threshold
    self.num_groundtruth_classes = num_groundtruth_classes

  def compute_object_detection_metrics(
      self, detected_boxes, detected_scores, detected_class_labels,
      groundtruth_boxes, groundtruth_class_labels,
      groundtruth_is_difficult_list, groundtruth_is_group_of_list,
      detected_masks=None, groundtruth_masks=None):
    """Evaluates detections as being tp, fp or ignored from a single image.

    The evaluation is done in two stages:
     1. All detections are matched to non group-of boxes; true positives are
        determined and detections matched to difficult boxes are ignored.
     2. Detections that are determined as false positives are matched against
        group-of boxes and ignored if matched.

    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_labels: A integer numpy array of shape [N, 1], repreneting
          the class labels of the detected N object instances.
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag
      detected_masks: (optional) A uint8 numpy array of shape
        [N, height, width]. If not None, the metrics will be computed based
        on masks.
      groundtruth_masks: (optional) A uint8 numpy array of shape
        [M, height, width].

    Returns:
      scores: A list of C float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      tp_fp_labels: A list of C boolean numpy arrays. Each numpy array
          is of shape [K, 1], representing K True/False positive label of
          object instances detected with class label c
    """
    detected_boxes, detected_scores, detected_class_labels, detected_masks = (
        self._remove_invalid_boxes(detected_boxes, detected_scores,
                                   detected_class_labels, detected_masks))

    ## in order to replace the class label of the detection box, the things should happen here:
    ## replace detected_class_labels with GT labels of the box with the highest IoU: line below
    # detected_class_labels = detected_class_corrector(detected_boxes, detected_class_labels, detected_scores, groundtruth_boxes, groundtruth_class_labels, groundtruth_is_group_of_list)
    # detected_scores = detected_class_corrector(detected_boxes, detected_class_labels, detected_scores, groundtruth_boxes, groundtruth_class_labels, groundtruth_is_group_of_list)
    # print("------------------------------------------------")
    # print("length of detected_boxes: ", len(detected_boxes))
    # print("length of detected_scores: ", len(detected_scores))
    # print("length of gt_boxes: ", len(groundtruth_boxes))
    # print("length of detected_class_labels: ", len(detected_class_labels))
    # print("length of groundtruth_class_labels: ", len(groundtruth_class_labels))
    # print("detected_class_labels: ", detected_class_labels)
    # counter = {}
    # for label in detected_class_labels:
    #   if label in counter.keys():
    #     counter[label] += 1
    #   else:
    #     counter[label] = 1
    # print(counter)
    scores, tp_fp_labels = self._compute_tp_fp(
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detected_class_labels,
        groundtruth_boxes=groundtruth_boxes,
        groundtruth_class_labels=groundtruth_class_labels,
        groundtruth_is_difficult_list=groundtruth_is_difficult_list,
        groundtruth_is_group_of_list=groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)

    if 0:
      d_small = []
      d_medium = []
      d_big = []
      g_small = []
      g_medium = []
      g_big = []

      d_w = detected_boxes[:,2]-detected_boxes[:,0]
      d_h = detected_boxes[:,3]-detected_boxes[:,1]
      d_a = 1.306*d_w*d_h

      for i in range(d_a.shape[0]):
        if d_a[i] >= 7**2 and d_a[i]<= 32**2:
          d_small.append(i)
        elif d_a[i]>32**2 and d_a[i]<= 96**2:
          d_medium.append(i)
        elif d_a[i] >96**2:
          d_big.append(i)

      g_w = groundtruth_boxes[:,2]-groundtruth_boxes[:,0]
      g_h = groundtruth_boxes[:,3]-groundtruth_boxes[:,1]
      g_a = 1.306*g_w*g_h

      for j in range(g_a.shape[0]):
        if g_a[j] >= 7**2 and g_a[j]<= 32**2:
          g_small.append(j)
        elif g_a[j]>32**2 and g_a[j]<= 96**2:
          g_medium.append(j)
        elif g_a[j] >96**2:
          g_big.append(j)


      score_small, tp_fp_labels_small = self._compute_tp_fp(
          detected_boxes=detected_boxes[d_small],
          detected_scores=detected_scores[d_small],
          detected_class_labels=detected_class_labels[d_small],
          groundtruth_boxes=groundtruth_boxes[g_small],
          groundtruth_class_labels=groundtruth_class_labels[g_small],
          groundtruth_is_difficult_list=groundtruth_is_difficult_list[g_small],
          groundtruth_is_group_of_list=groundtruth_is_group_of_list[g_small],
          detected_masks=detected_masks,
          groundtruth_masks=groundtruth_masks)
      score_medium, tp_fp_labels_medium = self._compute_tp_fp(
          detected_boxes=detected_boxes[d_medium],
          detected_scores=detected_scores[d_medium],
          detected_class_labels=detected_class_labels[d_medium],
          groundtruth_boxes=groundtruth_boxes[g_medium],
          groundtruth_class_labels=groundtruth_class_labels[g_medium],
          groundtruth_is_difficult_list=groundtruth_is_difficult_list[g_medium],
          groundtruth_is_group_of_list=groundtruth_is_group_of_list[g_medium],
          detected_masks=detected_masks,
          groundtruth_masks=groundtruth_masks)
      score_big, tp_fp_labels_big = self._compute_tp_fp(
          detected_boxes=detected_boxes[d_big],
          detected_scores=detected_scores[d_big],
          detected_class_labels=detected_class_labels[d_big],
          groundtruth_boxes=groundtruth_boxes[g_big],
          groundtruth_class_labels=groundtruth_class_labels[g_big],
          groundtruth_is_difficult_list=groundtruth_is_difficult_list[g_big],
          groundtruth_is_group_of_list=groundtruth_is_group_of_list[g_big],
          detected_masks=detected_masks,
          groundtruth_masks=groundtruth_masks)

    #print(score_small, tp_fp_labels_small, score_medium, tp_fp_labels_medium, score_big, tp_fp_labels_big)
    #exit()
    return scores, tp_fp_labels #scores, tp_fp_labels

  def _compute_tp_fp(self, detected_boxes, detected_scores,
                     detected_class_labels, groundtruth_boxes,
                     groundtruth_class_labels, groundtruth_is_difficult_list,
                     groundtruth_is_group_of_list,
                     detected_masks=None, groundtruth_masks=None):
    """Labels true/false positives of detections of an image across all classes.

    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_labels: A integer numpy array of shape [N, 1], representing
          the class labels of the detected N object instances.
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag
      detected_masks: (optional) A np.uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A np.uint8 numpy array of shape
        [M, height, width].

    Returns:
      result_scores: A list of float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
          shape [K, 1], representing K True/False positive label of object
          instances detected with class label c

    Raises:
      ValueError: If detected masks is not None but groundtruth masks are None,
        or the other way around.
    """
    if detected_masks is not None and groundtruth_masks is None:
      raise ValueError(
          'Detected masks is available but groundtruth masks is not.')
    if detected_masks is None and groundtruth_masks is not None:
      raise ValueError(
          'Groundtruth masks is available but detected masks is not.')

    result_scores = []
    result_tp_fp_labels = []
    for i in range(self.num_groundtruth_classes):
      groundtruth_is_difficult_list_at_ith_class = (
          groundtruth_is_difficult_list[groundtruth_class_labels == i])
      groundtruth_is_group_of_list_at_ith_class = (
          groundtruth_is_group_of_list[groundtruth_class_labels == i])

      (gt_boxes_at_ith_class, gt_masks_at_ith_class,
       detected_boxes_at_ith_class, detected_scores_at_ith_class,
       detected_masks_at_ith_class) = self._get_ith_class_arrays(
           detected_boxes, detected_scores, detected_masks,
           detected_class_labels, groundtruth_boxes, groundtruth_masks,
           groundtruth_class_labels, i)

      scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
          detected_boxes=detected_boxes_at_ith_class,
          detected_scores=detected_scores_at_ith_class,
          groundtruth_boxes=gt_boxes_at_ith_class,
          groundtruth_is_difficult_list=
          groundtruth_is_difficult_list_at_ith_class,
          groundtruth_is_group_of_list=
          groundtruth_is_group_of_list_at_ith_class,
          detected_masks=detected_masks_at_ith_class,
          groundtruth_masks=gt_masks_at_ith_class)

      result_scores.append(scores)
      result_tp_fp_labels.append(tp_fp_labels)
    return result_scores, result_tp_fp_labels

  def _get_overlaps_and_scores_box_mode(
      self,
      detected_boxes,
      detected_scores,
      groundtruth_boxes,
      groundtruth_is_group_of_list):
    """Computes overlaps and scores between detected and groudntruth boxes.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.

    Returns:
      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_group_of_boxlist.num_boxes() == 0 it will be None.
      scores: The score of the detected boxlist.
      num_boxes: Number of non-maximum suppressed detected boxes.
    """
    detected_boxlist = np_box_list.BoxList(detected_boxes)
    detected_boxlist.add_field('scores', detected_scores)
    gt_non_group_of_boxlist = np_box_list.BoxList(
        groundtruth_boxes[~groundtruth_is_group_of_list])
    iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
    scores = detected_boxlist.get_field('scores')
    num_boxes = detected_boxlist.num_boxes()
    return iou, None, scores, num_boxes

  def _compute_tp_fp_for_single_class(
      self, detected_boxes, detected_scores, groundtruth_boxes,
      groundtruth_is_difficult_list, groundtruth_is_group_of_list,
      detected_masks=None, groundtruth_masks=None):
    """Labels boxes detected with the same class from the same image as tp/fp.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not. If a
          groundtruth box is difficult, every detection matching this box
          is ignored.
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.
      detected_masks: (optional) A uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A uint8 numpy array of shape
        [M, height, width].

    Returns:
      Two arrays of the same size, containing all boxes that were evaluated as
      being true positives or false positives; if a box matched to a difficult
      box or to a group-of box, it is ignored.

      scores: A numpy array representing the detection scores.
      tp_fp_labels: a boolean numpy array indicating whether a detection is a
          true positive.
    """

    if detected_boxes.size == 0:
      return np.array([], dtype=float), np.array([], dtype=bool)

    # added by JJ
    #sorted_indices = np.argsort(detected_scores)
    #sorted_indices = sorted_indices[::-1]
    #detected_boxes = detected_boxes[sorted_indices, :]
    #detected_scores = detected_scores[sorted_indices]

    #print(detected_scores)
    #exit()


    (iou, _, scores,
     num_detected_boxes) = self._get_overlaps_and_scores_box_mode(
         detected_boxes=detected_boxes,
         detected_scores=detected_scores,
         groundtruth_boxes=groundtruth_boxes,
         groundtruth_is_group_of_list=groundtruth_is_group_of_list)

    if groundtruth_boxes.size == 0:
      return scores, np.zeros(num_detected_boxes, dtype=bool)
      # return np.zeros_like(scores), np.zeros(num_detected_boxes, dtype=bool)

    tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
    is_matched_to_difficult_box = np.zeros(num_detected_boxes, dtype=bool)
    is_matched_to_group_of_box = np.zeros(num_detected_boxes, dtype=bool)

    # The evaluation is done in two stages:
    # 1. All detections are matched to non group-of boxes; true positives are
    #    determined and detections matched to difficult boxes are ignored.
    # 2. Detections that are determined as false positives are matched against
    #    group-of boxes and ignored if matched.

    # Tp-fp evaluation for non-group of boxes (if any).
    if iou.shape[1] > 0:
      groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
          ~groundtruth_is_group_of_list]
      max_overlap_gt_ids = np.argmax(iou, axis=1) # iou: num detected boxes(of certain class) x num gt boxes
      is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
      for i in range(num_detected_boxes):
        gt_id = max_overlap_gt_ids[i] # detected_box_i indicates gt_id
        if iou[i, gt_id] >= self.matching_iou_threshold:
          if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
            if not is_gt_box_detected[gt_id]:
              tp_fp_labels[i] = True
              is_gt_box_detected[gt_id] = True
          else:
            is_matched_to_difficult_box[i] = True
        # else:
        #   scores[i] = 0 # no box - no score

    return scores[~is_matched_to_difficult_box
                  & ~is_matched_to_group_of_box], tp_fp_labels[
                      ~is_matched_to_difficult_box
                      & ~is_matched_to_group_of_box]

  def _get_ith_class_arrays(self, detected_boxes, detected_scores,
                            detected_masks, detected_class_labels,
                            groundtruth_boxes, groundtruth_masks,
                            groundtruth_class_labels, class_index):
    """Returns numpy arrays belonging to class with index `class_index`.

    Args:
      detected_boxes: A numpy array containing detected boxes.
      detected_scores: A numpy array containing detected scores.
      detected_masks: A numpy array containing detected masks.
      detected_class_labels: A numpy array containing detected class labels.
      groundtruth_boxes: A numpy array containing groundtruth boxes.
      groundtruth_masks: A numpy array containing groundtruth masks.
      groundtruth_class_labels: A numpy array containing groundtruth class
        labels.
      class_index: An integer index.

    Returns:
      gt_boxes_at_ith_class: A numpy array containing groundtruth boxes labeled
        as ith class.
      gt_masks_at_ith_class: A numpy array containing groundtruth masks labeled
        as ith class.
      detected_boxes_at_ith_class: A numpy array containing detected boxes
        corresponding to the ith class.
      detected_scores_at_ith_class: A numpy array containing detected scores
        corresponding to the ith class.
      detected_masks_at_ith_class: A numpy array containing detected masks
        corresponding to the ith class.
    """
    selected_groundtruth = (groundtruth_class_labels == class_index)
    gt_boxes_at_ith_class = groundtruth_boxes[selected_groundtruth]
    if groundtruth_masks is not None:
      gt_masks_at_ith_class = groundtruth_masks[selected_groundtruth]
    else:
      gt_masks_at_ith_class = None


    selected_detections = (detected_class_labels == class_index)
    detected_boxes_at_ith_class = detected_boxes[selected_detections]
    detected_scores_at_ith_class = detected_scores[selected_detections]
    if detected_masks is not None:
      detected_masks_at_ith_class = detected_masks[selected_detections]
    else:
      detected_masks_at_ith_class = None
    return (gt_boxes_at_ith_class, gt_masks_at_ith_class,
            detected_boxes_at_ith_class, detected_scores_at_ith_class,
            detected_masks_at_ith_class)

  def _remove_invalid_boxes(self, detected_boxes, detected_scores,
                            detected_class_labels, detected_masks=None):
    """Removes entries with invalid boxes.

    A box is invalid if either its xmax is smaller than its xmin, or its ymax
    is smaller than its ymin.

    Args:
      detected_boxes: A float numpy array of size [num_boxes, 4] containing box
        coordinates in [ymin, xmin, ymax, xmax] format.
      detected_scores: A float numpy array of size [num_boxes].
      detected_class_labels: A int32 numpy array of size [num_boxes].
      detected_masks: A uint8 numpy array of size [num_boxes, height, width].

    Returns:
      valid_detected_boxes: A float numpy array of size [num_valid_boxes, 4]
        containing box coordinates in [ymin, xmin, ymax, xmax] format.
      valid_detected_scores: A float numpy array of size [num_valid_boxes].
      valid_detected_class_labels: A int32 numpy array of size
        [num_valid_boxes].
      valid_detected_masks: A uint8 numpy array of size
        [num_valid_boxes, height, width].
    """
    valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2],
                                   detected_boxes[:, 1] < detected_boxes[:, 3])
    detected_boxes = detected_boxes[valid_indices]
    detected_scores = detected_scores[valid_indices]
    detected_class_labels = detected_class_labels[valid_indices]
    if detected_masks is not None:
      detected_masks = detected_masks[valid_indices]
    return [
        detected_boxes, detected_scores, detected_class_labels, detected_masks
    ]


