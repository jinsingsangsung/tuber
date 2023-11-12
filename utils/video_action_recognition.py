# pylint: disable=line-too-long
"""
Utility functions for task
"""
import glob
import json
import os
import time
import numpy as np

import torch
import math

from .utils import AverageMeter, accuracy, calculate_mAP, read_labelmap, print_log
from evaluates.evaluate_ava import STDetectionEvaluater, STDetectionEvaluaterSinglePerson
from evaluates.evaluate_ucf import STDetectionEvaluaterUCF
from evaluates.evaluate_jhmdb import STDetectionEvaluaterJHMDB

import requests
import traceback

__all__ = ["merge_jsons", "train_classification", "train_tuber_detection", "validate_tuber_detection", "validate_tuber_ucf_detection"]

def merge_jsons(result_dict, key, output_arr, gt_arr):
    if key not in result_dict.keys():
        result_dict[key] = {"preds": output_arr, "gts": gt_arr}
    else:
        result_dict[key]["preds"] = [max(*l) for l in zip(result_dict[key]["preds"], output_arr)]
        result_dict[key]["gts"] = [max(*l) for l in zip(result_dict[key]["gts"], gt_arr)]

def train_classification(base_iter, model, dataloader, epoch, criterion,
                         optimizer, cfg, writer=None):
    """Task of training video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    # batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='train_classification') 

    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1

        train_batch = data[0].cuda()
        train_label = data[1].cuda()
        data_time.update(time.time() - end)
        edd = time.time()
        # print("io: ", edd - end)

        end = edd
        outputs = model(train_batch)
        edd = time.time()
        # print("model: ", edd - end)
        loss = criterion(outputs, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_label.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        # batch_bar.set_postfix(
        #     loss="{:.04f}".format(float(loss / (step+1))),
        #     lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        # )
        # batch_bar.update()
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            iteration = base_iter
            # writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            # writer.add_scalar('train_batch_size_iteration', train_label.size(0), iteration)
            # writer.add_scalar('learning_rate', lr, iteration)
    # batch_bar.close()
    return base_iter

def train_tuber_detection(cfg, model, criterion, data_loader, optimizer, epoch, max_norm, lr_scheduler, scaler=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()
    model.train()
    criterion.train()
    save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    # header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10
    # batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='train_detection')
    for idx, data in enumerate(data_loader):

        data_time.update(time.time() - end)

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]
        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)
        for t in targets: del t["image_id"]

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.CONFIG.AMP):
            if cfg.CONFIG.TWO_STREAM:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, samples2, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, samples2, lfb_features)
                else:
                    outputs = model(samples, samples2)
            else:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, lfb_features)
                else:
                    if not "DN" in cfg.CONFIG.LOG.RES_DIR:
                        outputs = model(samples)
                        loss_dict = criterion(outputs, targets)
                    else:
                        dn_args = targets, cfg.CONFIG.MODEL.SCALAR, cfg.CONFIG.MODEL.LABEL_NOISE_SCALE, cfg.CONFIG.MODEL.BOX_NOISE_SCALE, cfg.CONFIG.MODEL.NUM_PATTERNS
                        outputs, mask_dict = model(samples, dn_args)
                        loss_dict = criterion(outputs, targets, mask_dict)       

        # if not math.isfinite(outputs["pred_logits"][0].data.cpu().numpy()[0,0]):
        #     print(outputs["pred_logits"][0].data.cpu().numpy())
        
        weight_dict = criterion.weight_dict
        if epoch > cfg.CONFIG.LOSS_COFS.WEIGHT_CHANGE:
            weight_dict['loss_ce'] = cfg.CONFIG.LOSS_COFS.LOSS_CHANGE_COF

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if (k in weight_dict and "bbox" not in k and "giou" not in k))
        # loss_dict.keys(): dict_keys(['loss_ce', 'loss_ce_b', 'class_error', 'loss_bbox', 'loss_giou'])  
        if scaler is None:
            optimizer.zero_grad()
            # optimizer_c.zero_grad()
            losses.backward()
            if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            min_scale = 128
            if scaler._scale < min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler._scale)
        # optimizer_c.step()
        lr_scheduler.step_update(epoch * len(data_loader) + idx)
        batch_time.update(time.time() - end)
        end = time.time()
        # batch_bar.set_postfix(
        #     loss="{:.04f}".format(float(loss / (step+1))),
        #     lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        # )
        # batch_bar.update()

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = '(train) Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print_log(save_path, print_string)
                for param in optimizer.param_groups:
                    lr = param['lr']
                print_log(save_path, 'lr: ', lr)

                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print_log(save_path, print_string)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            # print('length', len(targets))

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                try:
                    losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))
                except:
                    pass

            if not math.isfinite(loss_value):
                print_log(save_path, "Loss is {}, stopping training".format(loss_value))
                print_log(save_path, loss_dict_reduced)
                exit(1)

            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg,
                )
                print_log(save_path, print_string)

            # writer.add_scalar('train/class_error', class_err.avg, idx + epoch * len(data_loader))
            # writer.add_scalar('train/totall_loss', losses_avg.avg, idx + epoch * len(data_loader))
            # writer.add_scalar('train/loss_bbox', losses_box.avg, idx + epoch * len(data_loader))
            # writer.add_scalar('train/loss_giou', losses_giou.avg, idx + epoch * len(data_loader))
            # writer.add_scalar('train/loss_ce', losses_ce.avg, idx + epoch * len(data_loader))
            # writer.add_scalar('train/loss_ce_b', losses_ce_b.avg, idx + epoch * len(data_loader))
    
    try:
        metrics_data = json.dumps({
            '@epoch': epoch,
            '@step': epoch, # actually epoch
            '@time': time.time(),
            'class_error': float(class_err.avg),
            'loss': float(losses_avg.avg),
            'loss_giou': float(losses_giou.avg),
            'loss_ce': float(losses_ce.avg),
            # 'loss_ce_b': losses_ce_b.avg,
            })
        # Report JSON data to the NSML metric API server with a simple HTTP POST request.
        requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
    except requests.exceptions.RequestException:
        # Sometimes, the HTTP request might fail, but the training process should not be stopped.
        traceback.print_exc()
    # batch_bar.close()

@torch.no_grad()
def validate_tuber_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print_log(save_path, "remove {}".format(tmp_dir))
        print_log(save_path, "all tmp files removed")

    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                try:
                    model.training=False
                except:
                    pass
                if not "DN" in cfg.CONFIG.LOG.EXP_NAME:
                    outputs = model(samples)
                else:
                    dn_args = targets, cfg.CONFIG.MODEL.NUM_PATTERNS
                    outputs, mask_dict = model(samples, dn_args)
                    loss_dict = criterion(outputs, targets, mask_dict)
                # outputs, num_boxes_per_batch_idx = model(targets, samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        if not "sparse" in cfg.CONFIG.LOG.RES_DIR:
            try:
                scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
            except:
                scores, boxes = postprocessors['bbox'](outputs, orig_target_sizes)
        else: 
            scores, _ = postprocessors['bbox'](outputs, orig_target_sizes)
            boxes = outputs["pred_boxes"].detach().cpu().numpy()
        
        for bidx in range(scores.shape[0]): 
            # batch_pointer = 0
            # if bidx >= num_boxes_per_batch_idx[batch_pointer]:
                # batch_pointer += 1
            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]
            # frame_id = batch_id[batch_pointer][0]
            # key_pos = batch_id[batch_pointer][1]

            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                out_key_pos = key_pos // cfg.CONFIG.MODEL.DS_RATE

                buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                try:
                    buff_binary.append(output_b[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                except:
                    pass
            else:
                buff_output.append(scores[bidx])
                buff_anno.append(boxes[bidx])
                try:
                    buff_binary.append(output_b[bidx])
                except:
                    pass

            for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                buff_id.extend([frame_id])

            # raw_idx = (targets[bidx]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()
            raw_idx = torch.nonzero(targets[bidx]["raw_boxes"][:, 1] == key_pos, as_tuple=False).squeeze()
            # raw_idx = (targets[batch_pointer]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()

            val_label = targets[bidx]["labels"][raw_idx]
            # val_label = targets[batch_pointer]["labels"][raw_idx]
            val_label = val_label.reshape(-1, val_label.shape[-1])
            raw_boxes = targets[bidx]["raw_boxes"][raw_idx]
            # raw_boxes = targets[batch_pointer]["raw_boxes"][raw_idx]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])
            # print('raw_boxes',raw_boxes.shape)

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())


            # print(buff_anno, buff_GT_anno)

            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]

            buff_GT_id.extend(img_id_item)
            

        batch_time.update(time.time() - end)
        end = time.time()

        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = '(val) Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print_log(save_path, print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print_log(save_path, print_string)

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                try:
                    losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))
                except:
                    pass

            if not math.isfinite(loss_value):
                print_log(save_path, "Loss is {}, stopping eval".format(loss_value))
                print_log(save_path, loss_dict_reduced)
                exit(1)
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg,
                    # loss_ce_b=losses_ce_b.avg,
                    # cardinality_error=loss_dict_reduced['cardinality_error']
                )
                print_log(save_path, print_string)
            # print("len(buff_id): ", len(buff_id))
            # print("len(buff_binary): ", len(np.concatenate(buff_binary, axis=0)))
            # print("len(buff_anno): ", len(np.concatenate(buff_anno, axis=0)))
            # print("len(buff_output): ", len(np.concatenate(buff_output, axis=0)))
            # print("len(buff_GT_id): ", len(buff_GT_id))
            # print("len(buff_GT_label): ", len(np.concatenate(buff_GT_label, axis=0)))
            # print("len(buff_GT_anno): ", len(np.concatenate(buff_GT_anno, axis=0)))

    # if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
    #     writer.add_scalar('val/class_error', class_err.avg, epoch)
    #     writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
    #     writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
    #     writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
    #     writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)
    #     writer.add_scalar('val/loss_ce_b', losses_ce_b.avg, epoch)

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    try:
        buff_binary = np.concatenate(buff_binary, axis=0)
    except:
        pass
    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    # print(buff_output.shape, buff_anno.shape, buff_binary.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))
    
    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            try:
                data = np.concatenate([buff_anno[x], buff_output[x], buff_binary[x]])
            except:
                data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))
    print_log(save_path, "tmp files are all loaded")

    # write files and align all workers
    torch.distributed.barrier()
    # aggregate files
    Map_ = 0
    # aggregate files
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # read results
        if cfg.CONFIG.VAL.PUT_GT:
            evaluater = STDetectionEvaluater(cfg.CONFIG.DATA.LABEL_PATH, tiou_thresholds=[1e-8], class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        else:
            evaluater = STDetectionEvaluater(cfg.CONFIG.DATA.LABEL_PATH, tiou_thresholds=[0.5], class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print_log(save_path, metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print_log(save_path, print_string)
        print_log(save_path, mAP)
        # writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]

    if Map_ != 0:
        metrics_data = json.dumps({
                '@epoch': epoch,
                '@step': epoch, # actually epoch
                '@time': time.time(),
                'val_class_error': class_err.avg,
                'val_loss': losses_avg.avg,
                'val_loss_giou': losses_giou.avg,
                'val_loss_ce': losses_ce.avg,
                # 'val_loss_ce_b': losses_ce_b.avg,
                'val_mAP': Map_
                })
        try:
            # Report JSON data to the NSML metric API server with a simple HTTP POST request.
            requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
        except requests.exceptions.RequestException:
            # Sometimes, the HTTP request might fail, but the training process should not be stopped.
            traceback.print_exc()    
    torch.distributed.barrier()
    time.sleep(30)
    return Map_

@torch.no_grad()
def validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print_log(save_path, "remove {}".format(tmp_dir))
        print_log(save_path, "all tmp files removed")

    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                try:
                    model.training=False
                except:
                    pass
                if not "DN" in cfg.CONFIG.LOG.EXP_NAME:
                    outputs = model(samples)
                else:
                    dn_args = targets, cfg.CONFIG.MODEL.NUM_PATTERNS
                    outputs, mask_dict = model(samples, dn_args)
                    loss_dict = criterion(outputs, targets, mask_dict)

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        try:
            scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
        except:
            scores, boxes = postprocessors['bbox'](outputs, orig_target_sizes)

        B = scores.shape[0]
        T = scores.shape[1]
        scores = scores.reshape(-1, *scores.shape[-2:])
        boxes = boxes.reshape(-1, *boxes.shape[-2:])

        for bidx in range(B):

            if len(targets[bidx]["raw_boxes"]) == 0:
                continue

            frame_id = batch_id[bidx][0]
            # key_pos = batch_id[bidx][1]

            # out_key_pos = key_pos // cfg.CONFIG.MODEL.DS_RATE
            # out_key_pos = key_pos
            # print("key pos: {}, ds_rate: {}".format(key_pos, cfg.CONFIG.MODEL.DS_RATE))
            # scores: BT x num_q x num_c
            front_pad = targets[bidx]["front_pad"]
            end_pad = targets[bidx]["end_pad"]
            buff_output.append(scores[bidx*T+front_pad:(bidx+1)*T-end_pad, :, :].reshape(-1, scores.shape[-1]))
            buff_anno.append(boxes[bidx*T+front_pad:(bidx+1)*T-end_pad, :, :].reshape(-1, boxes.shape[-1]))
            try:
                buff_binary.append(output_b)
            except:
                pass

            for t in range(T-front_pad-end_pad):
                buff_GT_id.extend([frame_id + f"_{t:02d}"])
                for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                    buff_id.extend([frame_id + f"_{t:02d}"])
                    try:
                        buff_binary.append(output_b[..., 0])
                    except:
                        pass

            val_label = targets[bidx]["labels"] # length T
            # make one-hot vector
            val_category = torch.full((len(val_label), cfg.CONFIG.DATA.NUM_CLASSES+1), 0)
            for vl in range(len(val_label)):
                label = int(val_label[vl])
                val_category[vl, label] = 1
            val_label = val_category[front_pad:T-end_pad]

            raw_boxes = targets[bidx]["raw_boxes"]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])[front_pad:T-end_pad]

            # print('raw_boxes',raw_boxes.shape)

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())
            # img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in range(len(raw_boxes))]
            # buff_GT_id.extend(img_id_item)

        batch_time.update(time.time() - end)
        end = time.time()

        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = '(val) Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print_log(save_path, print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print_log(save_path, print_string)

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if not math.isfinite(loss_value):
                print_log(save_path, "Loss is {}, stopping eval".format(loss_value))
                print_log(save_path, loss_dict_reduced)
                exit(1)
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg
                )
                print_log(save_path, print_string)

    # if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
    #     writer.add_scalar('val/class_error', class_err.avg, epoch)
    #     writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
    #     writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
    #     writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
    #     writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    # try:
    #     buff_binary = np.concatenate(buff_binary, axis=0)
    # except:
    #     pass

    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    # print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))
    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    # try:
    #     tmp_binary_path = '{}/{}/binary_{}.txt'
    #     with open(tmp_binary_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
    #         for x in range(len(buff_id)):
    #             data = buff_binary[x]
    #             f.write("{} {}\n".format(buff_id[x], data.tolist()))
    # except:
    #     pass

    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    torch.distributed.barrier()
    # aggregate files
    Map_ = 0
    Map_v = 0
    # aggregate files
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # read results
        if cfg.CONFIG.DATA.DATASET_NAME == "ucf":
            evaluater = STDetectionEvaluaterUCF(class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        elif cfg.CONFIG.DATA.DATASET_NAME == "jhmdb":
            evaluater = STDetectionEvaluaterJHMDB(class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics, v_mAP, v_metrics = evaluater.evaluate()
        print_log(save_path, metrics)
        print_string = 'f-mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print_log(save_path, print_string)
        print_log(save_path, mAP)
        print_log(save_path, "video-level eval")
        print_log(save_path, v_metrics)
        print_string = 'v-mAP: {v_mAP:.5f}'.format(v_mAP=v_mAP[0])
        print_log(save_path, print_string)
        print_log(save_path, v_mAP)
        # writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]
        Map_v = v_mAP[0]
    if Map_ != 0:
        metrics_data = json.dumps({
                '@epoch': epoch,
                '@step': epoch, # actually epoch
                '@time': time.time(),
                'val_class_error': float(class_err.avg),
                'val_loss': float(losses_avg.avg),
                'val_loss_giou': float(losses_giou.avg),
                'val_loss_ce': float(losses_ce.avg),
                # 'val_loss_ce_b': losses_ce_b.avg,
                'val_mAP': Map_,
                'val_v_mAP': Map_v
                })
        try:
            # Report JSON data to the NSML metric API server with a simple HTTP POST request.
            requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
        except requests.exceptions.RequestException:
            # Sometimes, the HTTP request might fail, but the training process should not be stopped.
            traceback.print_exc()  

    torch.distributed.barrier()
    return Map_, Map_v

@torch.no_grad()
def validate_tuber_ucf_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print_log(save_path, "remove {}".format(tmp_dir))
        print_log(save_path, "all tmp files removed")
    
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                try:
                    model.training=False
                except:
                    pass
                if not "DN" in cfg.CONFIG.LOG.EXP_NAME:
                    outputs = model(samples)
                else:
                    dn_args = targets, cfg.CONFIG.MODEL.NUM_PATTERNS
                    outputs, mask_dict = model(samples, dn_args)
                    loss_dict = criterion(outputs, targets, mask_dict)

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        try:
            scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
        except:
            scores, boxes = postprocessors['bbox'](outputs, orig_target_sizes)

        B = scores.shape[0]
        T = scores.shape[1]
        scores = scores.reshape(-1, *scores.shape[-2:])
        boxes = boxes.reshape(-1, *boxes.shape[-2:])
        
        for bidx in range(B):

            if len(targets[bidx]["raw_boxes"]) == 0:
                continue

            frame_id = batch_id[bidx][0]
            # key_pos = batch_id[bidx][1]
            # out_key_pos = key_pos // cfg.CONFIG.MODEL.DS_RATE
            # out_key_pos = key_pos
            # print("key pos: {}, ds_rate: {}".format(key_pos, cfg.CONFIG.MODEL.DS_RATE))
            # scores: BT x num_q x num_c
            front_pad = targets[bidx]["front_pad"]
            end_pad = targets[bidx]["end_pad"]

            buff_output.append(scores[bidx*T+front_pad:(bidx+1)*T-end_pad, :, :].reshape(-1, scores.shape[-1]))
            buff_anno.append(boxes[bidx*T+front_pad:(bidx+1)*T-end_pad, :, :].reshape(-1, boxes.shape[-1]))
            
            val_label = targets[bidx]["labels"] # num_actors, length T
            # make one-hot vector
            
            val_category = torch.full((*val_label.shape[:2], cfg.CONFIG.DATA.NUM_CLASSES+1), 0)
            for i, vl in enumerate(val_label):
                for t in range(len(vl)):
                    label = int(vl[t])
                    val_category[i, t, label] = 1
            val_label = val_category[:, front_pad:T-end_pad, :]

            raw_boxes = targets[bidx]["raw_boxes"].reshape(-1, cfg.CONFIG.MODEL.TEMP_LEN, 6)
            raw_boxes = raw_boxes[:, front_pad:T-end_pad, :]

            buff_GT_label.append(val_label.transpose(0,1).flatten(0,1).detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.transpose(0,1).flatten(0,1).detach().cpu().numpy())
            # img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in range(len(raw_boxes))]
            # buff_GT_id.extend(img_id_item)
            # TODO :need to add number of instances as well
            num_boxes = targets[bidx]["boxes"].size(0) // cfg.CONFIG.MODEL.TEMP_LEN
            for t in range(T-front_pad-end_pad):
                frame_idx = int(buff_GT_anno[B*idx+bidx][num_boxes*t][1])
                buff_GT_id.extend([frame_id + f"_{frame_idx:03d}"] * num_boxes)
                for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                    buff_id.extend([frame_id + f"_{frame_idx:03d}"])

        batch_time.update(time.time() - end)
        end = time.time()

        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = '(val) Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print_log(save_path, print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print_log(save_path, print_string)

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if not math.isfinite(loss_value):
                print_log(save_path, "Loss is {}, stopping eval".format(loss_value))
                print_log(save_path, loss_dict_reduced)
                exit(1)
            if idx % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg
                )
                print_log(save_path, print_string)

    # if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
    #     writer.add_scalar('val/class_error', class_err.avg, epoch)
    #     writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
    #     writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
    #     writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
    #     writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    # try:
    #     buff_binary = np.concatenate(buff_binary, axis=0)
    # except:
    #     pass

    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    # print_log(save_path, buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))
    tmp_path = '{}/{}/{}.txt'

    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    # try:
    #     tmp_binary_path = '{}/{}/binary_{}.txt'
    #     with open(tmp_binary_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
    #         for x in range(len(buff_id)):
    #             data = buff_binary[x]
    #             f.write("{} {}\n".format(buff_id[x], data.tolist()))
    # except:
    #     pass

    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    torch.distributed.barrier()
    # aggregate files
    Map_ = 0
    Map_v = 0
    # aggregate files
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # read results
        if cfg.CONFIG.DATA.DATASET_NAME == "ucf":
            evaluater = STDetectionEvaluaterUCF(class_num=cfg.CONFIG.DATA.NUM_CLASSES, query_num=cfg.CONFIG.MODEL.QUERY_NUM)
        elif cfg.CONFIG.DATA.DATASET_NAME == "jhmdb":
            evaluater = STDetectionEvaluaterJHMDB(class_num=cfg.CONFIG.DATA.NUM_CLASSES, query_num=cfg.CONFIG.MODEL.QUERY_NUM)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics, v_mAP, v_metrics = evaluater.evaluate()
        # print_log(save_path, metrics)
        print_string = 'f-mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print_log(save_path, print_string)
        print_log(save_path, mAP)
        print_log(save_path, "video-level eval")
        # print_log(save_path, v_metrics)
        print_string = 'v-mAP: {v_mAP:.5f}'.format(v_mAP=v_mAP[0])
        print_log(save_path, print_string)
        print_log(save_path, v_mAP)
        # writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]
        Map_v = v_mAP[0]
    if Map_ != 0:
        metrics_data = json.dumps({
                '@epoch': epoch,
                '@step': epoch, # actually epoch
                '@time': time.time(),
                'val_class_error': float(class_err.avg),
                'val_loss': float(losses_avg.avg),
                'val_loss_giou': float(losses_giou.avg),
                'val_loss_ce': float(losses_ce.avg),
                # 'val_loss_ce_b': losses_ce_b.avg,
                'val_mAP': Map_,
                'val_v_mAP': Map_v
                })
        try:
            # Report JSON data to the NSML metric API server with a simple HTTP POST request.
            requests.post(os.environ['NSML_METRIC_API'], data=metrics_data)
        except requests.exceptions.RequestException:
            # Sometimes, the HTTP request might fail, but the training process should not be stopped.
            traceback.print_exc()  

    # torch.distributed.barrier()
    return Map_, Map_v
