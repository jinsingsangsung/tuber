import argparse
import datetime
import time

import torch
import torch.optim
from models.dab_conv_trans import build_model
from utils.model_utils_0919 import deploy_model, load_model, save_checkpoint, load_model_and_states
from utils.video_action_recognition import train_tuber_detection, validate_tuber_detection, validate_tuber_ucf_detection, validate_tuber_jhmdb_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir, print_log
from utils.lr_scheduler import build_scheduler
from utils.nsml_utils import *
import numpy as np
import random
import os

# Function to write or append the this_ip to the file
def write_this_ip_to_file(file_path, this_ip):
    with open(file_path, 'a') as file:
        file.write(this_ip + '\n')

# Function to read the file and return a list of IPs
def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        ip_list = file.read().splitlines()
    return ip_list

def main_worker(cfg):

    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # tb_logdir = build_log_dir(cfg)
        save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
        # writer = SummaryWriter(log_dir=tb_logdir)
        writer = None
    else:
        writer = None

    if int(os.getenv('NSML_SESSION', '0')) > 0:
        cfg.CONFIG.MODEL.LOAD = True
        cfg.CONFIG.MODEL.LOAD_FC = True
        cfg.CONFIG.MODEL.LOAD_DETR = False

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
    
    if cfg.CONFIG.AMP:
        scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
    else:
        scaler = None  

    study = os.environ["NSML_STUDY"]
    run = os.environ["NSML_RUN_NAME"].split("/")[-1]
    exp_name = cfg.CONFIG.LOG.EXP_NAME.format(study, run)

    if int(os.getenv('NSML_SESSION', '0')) > 0:
        # 실험 이어하기의 경우
        epochs_folder = os.listdir(os.path.join(cfg.CONFIG.LOG.BASE_PATH, exp_name, cfg.CONFIG.LOG.SAVE_DIR))
        epochs_folder.sort()
        latest_epoch = epochs_folder[-1]
        cfg.CONFIG.MODEL.PRETRAINED_PATH = os.path.join(cfg.CONFIG.LOG.BASE_PATH, exp_name, cfg.CONFIG.LOG.SAVE_DIR, latest_epoch) # find the pretrained_path
        model, optimizer, lr_scheduler, start_epoch = load_model_and_states(model, optimizer, lr_scheduler, scaler, cfg)
        cfg.CONFIG.TRAIN.START_EPOCH = start_epoch

    else:
        if cfg.CONFIG.MODEL.LOAD:
            model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Start training...')
    start_time = time.time()
    max_accuracy = 0.0  
    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if epoch > 0:
            last_epoch = os.path.join(cfg.CONFIG.LOG.BASE_PATH, exp_name, cfg.CONFIG.LOG.SAVE_DIR, f'ckpt_epoch_{epoch-1:02d}.pth')
        else: 
            last_epoch = ""
        if os.path.isfile(last_epoch):
            cfg.CONFIG.MODEL.LOAD = True
            cfg.CONFIG.MODEL.LOAD_FC = True
            cfg.CONFIG.MODEL.LOAD_DETR = False
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print_log(save_path, "bringing the previous epoch's .pth file...")
            cfg.CONFIG.MODEL.PRETRAINED_PATH = last_epoch
            model, optimizer, lr_scheduler, _ = load_model_and_states(model, optimizer, lr_scheduler, scaler, cfg)
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        train_tuber_detection(cfg, model, criterion, train_loader, optimizer, epoch, cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM, lr_scheduler, scaler, writer)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and (
                epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler)

        if (epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
                validate_tuber_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)
            elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
                validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)
            else:
                validate_tuber_ucf_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)
        if os.getenv('NSML_RANK', '0') == '0':
            set_nsml_reschedule()

    if writer is not None:
        writer.close()
    unset_nsml_reschedule()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='./configuration/Dab_conv_trans_CSN152_AVA22.yaml',
                        help='path to config file.')
    parser.add_argument('--random_seed', default=1, type=int, help='random_seed')
    parser.add_argument('--debug', action='store_true', help="debug, and ddp is disabled")
    parser.add_argument('--eff', action='store_true', help="only for AVA, efficiently output only keyframe")
    parser.add_argument('--grad_ckpt', action='store_true', help="use gradient checkpoint")
    parser.add_argument('--amp', action='store_true', help="use average mixed precision")
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    study = os.environ["NSML_STUDY"]
    run = os.environ["NSML_RUN_NAME"].split("/")[-1]

    cfg.CONFIG.LOG.RES_DIR = cfg.CONFIG.LOG.RES_DIR.format(study, run)
    cfg.CONFIG.LOG.EXP_NAME = cfg.CONFIG.LOG.EXP_NAME.format(study, run)
    if args.debug:
        cfg.DDP_CONFIG.DISTRIBUTED = False
        cfg.CONFIG.LOG.RES_DIR = "debug_{}-{}/res/".format(study,run)
        cfg.CONFIG.LOG.EXP_NAME = "debug_{}-{}".format(study,run)
    if args.eff:
        cfg.CONFIG.EFFICIENT = True
    else:
        cfg.CONFIG.EFFICIENT = False
    if args.grad_ckpt:
        cfg.CONFIG.GRADIENT_CHECKPOINTING = True     
    if args.amp:
        cfg.CONFIG.AMP = True          
    
    import socket 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    this_ip = s.getsockname()[0] # put this to world_url
    
    cfg.DDP_CONFIG.WORLD_SIZE = int(os.environ["NSML_WORLD_SIZE"])
    
    if cfg.DDP_CONFIG.WORLD_SIZE > 1:
        tmp_path = '{}/ip_lists/{}-{}.txt'
        file_path = tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, study, run)
        if not os.path.exists(file_path):    
            with open(file_path, 'w') as f:
                f.write(this_ip + '\n')
        else:
            write_this_ip_to_file(file_path, this_ip)
            
        while True:
            ip_lines = read_file_to_list(file_path)
            if len(ip_lines) == cfg.DDP_CONFIG.WORLD_SIZE:
                break
            time.sleep(0.5)
        
        ip_list = read_file_to_list(file_path)
        cfg.DDP_CONFIG.WOLRD_URLS = ip_list
        cfg.DDP_CONFIG.DIST_URL = cfg.DDP_CONFIG.DIST_URL.format(ip_list[0])        
        
    else:    
        cfg.DDP_CONFIG.DIST_URL = cfg.DDP_CONFIG.DIST_URL.format(this_ip)
        cfg.DDP_CONFIG.WOLRD_URLS[0] = cfg.DDP_CONFIG.WOLRD_URLS[0].format(this_ip)

    s.close()
    spawn_workers(main_worker, cfg)