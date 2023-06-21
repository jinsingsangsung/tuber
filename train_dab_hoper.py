import argparse
import datetime
import time

import torch
import torch.optim
# from tensorboardX import SummaryWriter
from models.dab_hoper import build_model
from utils.model_utils import deploy_model, load_model, save_checkpoint, load_model_and_states
from utils.video_action_recognition import train_tuber_detection, validate_tuber_detection, validate_tuber_ucf_detection, validate_tuber_jhmdb_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir, print_log
from utils.lr_scheduler import build_scheduler
from utils.nsml_utils import *
import numpy as np
import random
import os
from datetime import date
import wandb

def main_worker(cfg):

    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # tb_logdir = build_log_dir(cfg)
        save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
        # writer = SummaryWriter(log_dir=tb_logdir)
        writer = None
    else:
        writer = None

        # if int(os.getenv('NSML_SESSION', '0')) > 0:
        # cfg.CONFIG.MODEL.LOAD = True
        # cfg.CONFIG.MODEL.LOAD_FC = True
        # cfg.CONFIG.MODEL.LOAD_DETR = False

    # create model
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:    
        print_log(save_path, datetime.datetime.today())
        print_log(save_path, 'Creating TubeR model: %s' % cfg.CONFIG.MODEL.NAME)
        print_log(save_path, "use single frame:", cfg.CONFIG.MODEL.SINGLE_FRAME)
    model, criterion, postprocessors = build_model(cfg)
    model = deploy_model(model, cfg, is_tuber=True)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:    
        print_log(save_path, 'Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from datasets.ava_frame import build_dataloader
    elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        from datasets.jhmdb_frame_ import build_dataloader
    elif cfg.CONFIG.DATA.DATASET_NAME == 'ucf':
        from datasets.ucf_frame import build_dataloader        
    else:
        build_dataloader = None
        print("invalid dataset name")

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)

    # create criterion
    criterion = criterion.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and "class_embed" not in n and "query_embed" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "class_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR, #10
        },
        {
            "params": [p for n, p in model.named_parameters() if "query_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR, #10
        },
    ]

    # create optimizer
    if cfg.CONFIG.TRAIN.OPTIMIZER.NAME == 'ADAMW':
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.CONFIG.TRAIN.LR, weight_decay=cfg.CONFIG.TRAIN.W_DECAY)
    elif cfg.CONFIG.TRAIN.OPTIMIZER.NAME == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.CONFIG.TRAIN.LR, weight_decay=cfg.CONFIG.TRAIN.W_DECAY)
    else:
        raise AssertionError("optimizer is one of SGD or ADAMW")
    # create lr scheduler
    lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Start training...')
    start_time = time.time()
    max_accuracy = 0.0

    if cfg.CONFIG.LOG.WANDB and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        save_path = '/dataset/result/%s' % cfg.CONFIG.LOG.EXP_NAME
        prj_name = "_".join(cfg.CONFIG.LOG.EXP_NAME.split("_")[:2])
        exp_name = "_".join(cfg.CONFIG.LOG.EXP_NAME.split("_")[2:]) + datetime.datetime.now().strftime("%H:%M:%S")
        wandb.init(project=prj_name, name=exp_name)
        wandb.config.update(cfg)
        wandb.watch(model)

    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        train_tuber_detection(cfg, model, criterion, train_loader, optimizer, epoch, cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM, lr_scheduler, writer)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and (
                epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler)

        if (epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
                curr_accuracy = validate_tuber_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)
                if max_accuracy < curr_accuracy:
                    max_accuracy = curr_accuracy
                    os.remove(os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR,
                                  f'ckpt_epoch_{epoch:02d}.pth')
                                )
            elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
                validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)
            else:
                validate_tuber_ucf_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Training time {}'.format(total_time_str))
        print_log(save_path, 'Final performance: {}'.format(max_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='./configuration/Dab_hoper_CSN50_AVA22.yaml',
                        help='path to config file.')
    parser.add_argument('--random_seed', default=1, type=int, help='random_seed')
    parser.add_argument('--debug', action='store_true', help="debug, and ddp is disabled")
    parser.add_argument('--eff', action='store_true', help="only for AVA, efficiently output only keyframe")
    parser.add_argument('--use_cls_sa', action='store_true', help="attach self attention layer to the decoder")
    parser.add_argument('--rm_binary', action="store_true", help="remove binary branch")
    parser.add_argument('--cut_grad', action="store_true", help="cut cls loss gradient to the anchor box")
    parser.add_argument('--wandb', action="store_true", help="turn on the wandb")
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    datetime = date.today().strftime("%m/%d/%y")
    month = datetime.split("/")[0]
    day = datetime.split("/")[1]
    cfg.CONFIG.LOG.RES_DIR = cfg.CONFIG.LOG.RES_DIR.format(month, day)
    cfg.CONFIG.LOG.EXP_NAME = cfg.CONFIG.LOG.EXP_NAME.format(month, day)
    if args.debug:
        cfg.DDP_CONFIG.DISTRIBUTED = False
        cfg.CONFIG.LOG.RES_DIR = "debug_{}-{}/res/".format(month, day)
        cfg.CONFIG.LOG.EXP_NAME = "debug_{}-{}".format(month, day)
    if args.eff:
        cfg.CONFIG.EFFICIENT = True
    if args.use_cls_sa:
        cfg.CONFIG.MODEL.USE_CLS_SA = True
    if args.rm_binary:
        cfg.CONFIG.MODEL.RM_BINARY = True
    if args.cut_grad:
        cfg.CONFIG.TRAIN.CUT_GRADIENT = True
    if args.wandb:
        cfg.CONFIG.LOG.WANDB = True

    import socket 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    this_ip = s.getsockname()[0] # put this to world_url
    cfg.DDP_CONFIG.DIST_URL = cfg.DDP_CONFIG.DIST_URL.format(this_ip)
    cfg.DDP_CONFIG.WOLRD_URLS[0] = cfg.DDP_CONFIG.WOLRD_URLS[0].format(this_ip)
    s.close()
    spawn_workers(main_worker, cfg)
