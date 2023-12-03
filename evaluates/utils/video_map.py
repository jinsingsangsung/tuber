## copied and modified from https://github.com/wei-tim/YOWO/blob/86c240951fef53201cb3212b6539e9d3828ea41c/core/eval_results.py#L119

import numpy as np
import os
from evaluates.utils.video_map_utils import *

class VideoMAPEvaluator(object):

    def __init__(self, categories, matching_iou_threshold=0.5):
        self.categories = categories
        self.iou = matching_iou_threshold

    def add_gt(self, gt_videos):
        self.gt_videos = gt_videos
    
    def add_pred(self, all_boxes):
        self.all_boxes = all_boxes

    def evaluate_videoAP(self, bTemporal = True, prior_length=None):
        '''
        gt_videos: {vname:{tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}} 
        all_boxes: {imgname:{cls_ind:array[x1,y1,x2,y2, cls_score]}}
        '''
        CLASSES = ()
        for cls_dict in self.categories:
            CLASSES = CLASSES + (cls_dict["name"],)
        gt_videos = self.gt_videos
        all_boxes = self.all_boxes
        iou_thresh = self.iou
        if len(all_boxes.keys()) == 0:
            metrics = {}
            metrics["video-mAP@{}IOU".format(self.iou)] = 0
            print("need better result to compute video mAP")
            return metrics
        def imagebox_to_videts(img_boxes, CLASSES):
            # image names
            keys = list(img_boxes.keys())
            keys.sort()
            res = []
            # without 'background'
            for cls_ind, cls in enumerate(CLASSES[0:]):
                v_cnt = 1
                # frame_index = 1
                v_dets = []
                cls_ind += 1
                preVideo = "_".join(keys[0].split('_')[:-1])
                for i in range(len(keys)): #iterate through all frames
                    curVideo = "_".join(keys[i].split('_')[:-1])
                    frame_index = int(keys[i].split('_')[-1])
                    img_cls_dets = img_boxes[keys[i]][cls_ind]
                    # if len(img_cls_dets) == 0:
                        # import pdb; pdb.set_trace()
                    v_dets.append([frame_index, img_cls_dets])
                    # frame_index += 1
                    if preVideo!=curVideo:
                        preVideo = curVideo
                        # frame_index = 1
                        # tmp_dets = v_dets[-1]
                        del v_dets[-1]
                        res.append([cls_ind, v_cnt, v_dets])
                        v_cnt += 1
                        v_dets = []
                        # v_dets.append(tmp_dets)
                        v_dets.append([frame_index, img_cls_dets])
                        # frame_index += 1
                # the last video
                # print('num of videos:{}'.format(v_cnt))
                res.append([cls_ind, v_cnt, v_dets])
            return res

        gt_videos_format = gt_to_videts(gt_videos)
        pred_videos_format = imagebox_to_videts(all_boxes, CLASSES)
        ap_all = []
        metrics = {}
        for cls_ind, cls in enumerate(CLASSES[0:]):
            cls_ind += 1
            # [ video_index, [[frame_index, x1,y1,x2,y2]] ]
            gt = [g[1:] for g in gt_videos_format if g[0]==cls_ind]
            # len(pred_videos_format): class_num * num_videos
            # pred_videos_format[i]: list of length 3 where each element is:
            #     class_label: (i // num_classes) + 1
            #     video_idx: (i % num_classes) + 1
            #     list of n frames that has bounding boxes, confidence scores
            pred_cls = [p[1:] for p in pred_videos_format if p[0]==cls_ind]
            # if bTemporal: 
                # cls_len = prior_length[cls_ind]
            # else:
                # cls_len = None            
            cls_len = None
            ap = video_ap_one_class(gt, pred_cls, iou_thresh, bTemporal, cls_len)
            ap_all.append(ap)
            print("[v-mAP] ", cls, ap)
            metrics[cls] = ap
            # print(cls, metrics[cls])
        metrics["video-mAP@{}IOU".format(self.iou)] = np.mean(ap_all)
        return metrics


def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbox: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames

    return scores

def link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbox_list: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    # no detections at all
    if not ind_notempty:
        return []
    # miss some frames
    elif len(ind_notempty)!=nfr:     
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    detect = bbox_list
    nframes = len(detect)
    res = []
    isempty_vertex = np.zeros([nframes,], dtype=np.bool_)
    edge_scores = [compute_score_one_class(detect[i], detect[i+1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32) for d in detect]
        # viterbi
        # from the second last frame back
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)
            index[i] = np.argmax(edge_score, axis=1)
        # decode
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        # remove covered boxes and build output structures
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            if i < nframes-1: edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0: edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)
            isempty_vertex[i] = (detect[i].size==0) # it is true when there is no detection in any frame
        res.append( this )
        if len(res) == 3:
            break
        
    return res


def link_video_one_class(vid_det, bNMS3d = False, gtlen=None, start=None):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score]]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres = link_bbxes_between_frames(vdets) 
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [np.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            keep = nms_3d(dets, 0.3) # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                if gtlen and not start:
                    vres = temporal_check(vres_keep, gtlen)
                elif gtlen and start:
                    output_res = []
                    for v in vres_keep:
                        output_res.append(v[start:start+gtlen])
                    vres = output_res
                else:
                    vres = vres_keep

    return vres


def video_ap_one_class(gt, pred_videos, iou_thresh = 0.2, bTemporal = False, gtlen = None):
    '''
    gt: [ video_index, array[frame_index, x1,y1,x2,y2] ]
    pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []
    for pred_v in pred_videos:
        video_index = pred_v[0]
        valid_pred = [k for k in pred_v[1] if len(k[1])!=0]
        trim_len = len(valid_pred)
        try:
            trim_start = valid_pred[0][0]
        except:
            trim_start = None
        pred_link_v = link_video_one_class(pred_v[1], True, trim_len, trim_start) # [array<frame_index, x1,y1,x2,y2, cls_score>]
        for tube in pred_link_v:
            pred.append((video_index, tube))
    # sort tubes according to scores (descending order)
    argsort_scores = np.argsort(-np.array([np.mean(b[:, 5]) for _, b in pred])) 
    pr = np.empty((len(pred)+1, 2), dtype=np.float32) # precision, recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = len(gt) #sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    gt_v_index = [g[0] for g in gt]
    for i, k in enumerate(argsort_scores):
        # if i % 100 == 0:
        #     print ("%6.2f%% boxes processed, %d positives found, %d remain" %(100*float(i)/argsort_scores.size, tp, fn))
        video_index, boxes = pred[k]
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            for j, g in enumerate(gt):
                if g[0] == video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this) > 0:
                if bTemporal:
                    iou = np.array([iou3dt(np.array(g_), boxes[:, :5]) for g_ in gt_this])
                else:
                    iou = []
                    for g_ in gt_this:
                        if boxes.shape[0] > g_.shape[0]:
                            # in case some frame don't have gt 
                            iou.append(iou3d(g_, boxes[int(g_[0,0]-1):int(g_[-1,0]),:5]))
                            # try:
                            #     iou = np.array([iou3d(g_, boxes[int(g_[0,0]-1):int(g_[-1,0]),:5]) for g_ in gt_this]) 
                            # except:
                            #     print("case1")
                            #     print(gt_v_index)
                            #     print([g[0] for g in gt])
                            #     import pdb; pdb.set_trace()
                        elif boxes.shape[0]<g_.shape[0]:
                            # in flow case
                            iou.append(iou3d(g_[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]))
                            # try:
                            #     iou = np.array([iou3d(g[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]) for g in gt_this]) 
                            # except:
                            #     print("case2")
                            #     print(gt_v_index)
                            #     print([g[0] for g in gt])                            
                            #     import pdb; pdb.set_trace()
                        else:
                            iou.append(iou3d(g_, boxes[:,:5]))
                            # try:
                            #     iou = np.array([iou3d(g, boxes[:,:5]) for g in gt_this]) 
                            # except:
                            #     print(boxes.shape[0], g_.shape[0])
                            #     print([g[:, 0:] for g in gt_this])
                            #     print(boxes[:, 0])                         
                            #     print("case3")
                            #     import pdb; pdb.set_trace()
                    iou = np.array(iou)

                if iou.size > 0: # on ucf101 if invalid annotation ....
                    argmax = np.argmax(iou)
                    if iou[argmax] >= iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn + 0.00001)
    ap = voc_ap(pr)

    return ap

def get_max_subset(x_org, gtL):
    x = x_org - np.mean(x_org)
    bestSoFar = 0
    bestNow = 0
    bestStartIndexSoFar = -1
    bestStopIndexSoFar = -1
    bestStartIndexNow = -1
    for i in range(x.shape[0]):
        value = bestNow + x[i]
        if value > 0:
            if bestNow == 0:
                bestStartIndexNow = i
            bestNow = value
        else:
            bestNow = 0
        if bestNow > bestSoFar:
            bestSoFar = bestNow
            bestStopIndexSoFar = i
            bestStartIndexSoFar = bestStartIndexNow
#    # search suitable length surrounding: approximate method
#    L_d = bestStopIndexSoFar-bestStartIndexSoFar
#    lcost = - (|gt_L - L_d| / gt_L)
    if gtL>(bestStopIndexSoFar-bestStartIndexSoFar):
        ext = (gtL - (bestStopIndexSoFar-bestStartIndexSoFar))//2
        bestStartIndexSoFar -= ext
        bestStopIndexSoFar += ext      
    elif gtL<(bestStopIndexSoFar-bestStartIndexSoFar):
        ext = ((bestStopIndexSoFar-bestStartIndexSoFar) - gtL)//2
        bestStartIndexSoFar += ext
        bestStopIndexSoFar -= ext

    if bestStartIndexSoFar<0: bestStartIndexSoFar=0
    if bestStopIndexSoFar>x.shape[0]: bestStopIndexSoFar=x.shape[0]
    return bestSoFar, bestStartIndexSoFar, bestStopIndexSoFar

def temporal_check(tubes, gt_L):
    # nframes x 6 array <frame> <x1> <y1> <x2> <y2> <score>
    # objective: max ( mean(score[L_d]) - (|gt_L - L_d| / gt_L) )
    save_tubes = []
    for tube in tubes:  #bbiou = iou2d(d2[:,1:5],b1)
        nframes = tube.shape[0]
        edge_scores = np.array([iou2d(tube[i,1:5],tube[i+1,1:5]) for i in range(nframes-1)]) # +tube[i,5]
        # if both overlap and cls score are low, then reverse the score, they should be remove from the tube
        ind = np.where(edge_scores<0.3)[0] + 1  
        score = tube[:, 5]
        score[ind] = -score[ind]
        best_v, beststart, bestend = get_max_subset(score, gt_L)
        trimed_b = tube[int(beststart):int(bestend), :]
        save_tubes.append(trimed_b)
    return save_tubes

def gt_to_videts(gt_v):
    # return  [label, video_index, [[frame_index, x1,y1,x2,y2], [], []] ]
    keys = list(gt_v.keys())
    keys.sort()
    res = []
    for i in range(len(keys)):
        # annotation of the video: tubes and gt_classes
        v_annot = gt_v[keys[i]]
        for j in range(len(v_annot['tubes'])):
            res.append([v_annot['gt_classes'], i+1, v_annot['tubes'][j]])
        # res.append([v_annot['gt_classes'], i+1, v_annot['tubes']])
    return res


