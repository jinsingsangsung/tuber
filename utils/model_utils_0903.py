import os
import hashlib
import requests
from tqdm import tqdm
import torch.nn as nn
import random
import torch
import numpy as np
from .utils import print_log

__all__ = ['load_detr_weights', 'deploy_model', 'load_model', 'save_model', 'save_checkpoint', 'check_sha1', 'download']

def load_detr_weights(model, pretrain_dir, cfg):
    checkpoint = torch.load(pretrain_dir, map_location='cpu')
    model_dict = model.state_dict()
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    pretrained_dict = {}
    distributed = cfg.DDP_CONFIG.DISTRIBUTED
    # if "dab-d-tuber-detr" in cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR:
    #     l = 0
    # else:
    #     l = 1  
    l = 0      
    for k, v in checkpoint['model'].items():
        if k.split('.')[l] == 'transformer':
            if "offset_embed" in k:
                for i in range(6):
                    pretrained_dict.update({k.replace("offset_embed.layers", "offset_embed.{}.layers".format(i)): v})
            pretrained_dict.update({k: v})
        elif k.split('.')[l] == 'bbox_embed':
            pretrained_dict.update({k: v})
        elif k.split('.')[l] == 'query_embed':
            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM * (cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE) # 10 * 32 //8
                pretrained_dict.update({k:v[:query_size].repeat(cfg.CONFIG.MODEL.DS_RATE, 1)})
            else:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM
                pretrained_dict.update({k:v[:query_size]})
            # print(v.shape) # v의 len이 135임! 따라서 query size > 135면 error남
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print_log(log_path, "query_size:", query_size)
            # pretrained_dict.update({k: v[:query_size]})
        elif cfg.CONFIG.EFFICIENT and "refpoint_embed" in k:
            t = cfg.CONFIG.MODEL.TEMP_LEN
            nq = cfg.CONFIG.MODEL.QUERY_NUM
            try:
                if model_dict[k].shape[0] < checkpoint["model"][k].shape[0]:
                    v = v.reshape(t, nq, 4)[t//2]
            except:
                if model_dict[k[7:]].shape[0] < checkpoint["model"][k].shape[0]:
                    v = v.reshape(t, nq, 4)[t//2]
            pretrained_dict.update({k: v})
        elif "refpoint_embed" in k:
            t = cfg.CONFIG.MODEL.TEMP_LEN
            nq = cfg.CONFIG.MODEL.QUERY_NUM
            try:
                if model_dict[k].shape[0] < checkpoint["model"][k].shape[0]:
                    v = v[:nq]
            except:
                if model_dict["module."+k].shape[0] < checkpoint["model"][k].shape[0]:
                    v = v[:nq]
            pretrained_dict.update({k: v})
    if distributed:
        pretrained_dict_ = {"module."+k: v for k, v in pretrained_dict.items() if "module."+k in model_dict}
        unused_dict = {"module."+k: v for k, v in pretrained_dict.items() if not "module."+k in model_dict}
        not_found_dict = {k: v for k, v in model_dict.items() if not k[7:] in pretrained_dict}
    else:
        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        unused_dict = {k: v for k, v in pretrained_dict.items() if not k in model_dict}
        not_found_dict = {k: v for k, v in model_dict.items() if not k in pretrained_dict}

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print_log(log_path, "number of detr unused model layers:", len(unused_dict.keys()))
        print_log(log_path, "number of detr used model layers:", len(pretrained_dict_.keys()))
    # print("model_dict",[i for i in model_dict.keys()][:10])
        print_log(log_path, "not found layers:", len([k for k in not_found_dict.keys() if not "backbone" in k]))

    model_dict.update(pretrained_dict_)
    model.load_state_dict(model_dict)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        if len(pretrained_dict_.keys())!=0:
            print_log(log_path, "detr load pretrain success")
        else:
            print_log(log_path, "detr load pretrain failed")


def deploy_model(model, cfg, is_tuber=True):
    """
    Deploy model to multiple GPUs for DDP training.
    """
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if cfg.DDP_CONFIG.DISTRIBUTED:
        if cfg.DDP_CONFIG.GPU is not None:
            torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
            model.cuda(cfg.DDP_CONFIG.GPU)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[cfg.DDP_CONFIG.GPU],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.DDP_CONFIG.GPU is not None:
        torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
        model = model.cuda(cfg.DDP_CONFIG.GPU)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if cfg.CONFIG.MODEL.LOAD_DETR: #and is_tuber:  
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print_log(log_path, "loading detr")
            load_detr_weights(model, cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR, cfg)
    else:
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print_log(log_path, "detr is not loaded")

    return model


def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
        print_log(log_path, "=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        if cfg.DDP_CONFIG.GPU is None:
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()
        if not load_fc:
            del model_dict['module.fc.weight']
            del model_dict['module.fc.bias']
        # detr weight is already updated
        # del model_dict['module.query_embed.weight']
        # print(model_dict.keys())
        if cfg.DDP_CONFIG.DISTRIBUTED:
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            unused_dict = {k: v for k, v in checkpoint['model'].items() if not k in model_dict}
            not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint['model']}
            print_log(log_path,"number of unused model layers:", len(unused_dict.keys()))
            print_log(log_path,"number of not found layers:", len(not_found_dict.keys()))
            
        else:
            pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k[7:] in model_dict}
            unused_dict = {k: v for k, v in checkpoint["model"].items() if not k[7:] in model_dict}
            not_found_dict = {k: v for k, v in model_dict.items() if not "module."+k in checkpoint["model"]}
            print_log(log_path,"number of loaded model layers:", len(pretrained_dict.keys()))
            print_log(log_path,"number of unused model layers:", len(unused_dict.keys()))
            print_log(log_path,"number of not found layers:", len(not_found_dict.keys()))

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        try:
            print_log(log_path,"=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
        except:
            print_log(log_path,"=> loaded checkpoint '{}' (epoch 0)".format(cfg.CONFIG.MODEL.PRETRAINED_PATH, 0))
    else:
        print_log(log_path,"=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None

def load_model_and_states(model, optimizer, scheduler, cfg):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
        print_log(log_path, "=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        if cfg.DDP_CONFIG.GPU is None:
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()

        if cfg.DDP_CONFIG.DISTRIBUTED:
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            unused_dict = {k: v for k, v in checkpoint['model'].items() if not k in model_dict}
            not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint['model']}
            print_log(log_path,"number of unused model layers:", len(unused_dict.keys()))
            print_log(log_path,"number of not found layers:", len(not_found_dict.keys()))
            
        else:
            pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k[7:] in model_dict}
            unused_dict = {k: v for k, v in checkpoint["model"].items() if not k[7:] in model_dict}
            not_found_dict = {k: v for k, v in model_dict.items() if not "module."+k in checkpoint["model"]}
            print_log(log_path,"number of loaded model layers:", len(pretrained_dict.keys()))
            print_log(log_path,"number of unused model layers:", len(unused_dict.keys()))
            print_log(log_path,"number of not found layers:", len(not_found_dict.keys()))

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print_log(log_path,"=> loaded checkpoint '{}' (epoch {})".format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
    else:
        print_log(log_path,"=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
    
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    random.setstate(checkpoint["random_python"])
    np.random.set_state(checkpoint["random_numpy"])
    torch.set_rng_state(checkpoint['random_pytorch'].cpu())
    if model.device == 'cuda':
        torch.cuda.set_rng_state(checkpoint['random_cuda'])
    return model, optimizer, scheduler, start_epoch


def save_model(model, optimizer, epoch, cfg):
    # pylint: disable=line-too-long
    """
    Save trained model weights.
    """
    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    ckpt_name = "f{}_s{}_ckpt_epoch{}.pth".format(cfg.CONFIG.DATA.CLIP_LEN, cfg.CONFIG.DATA.FRAME_RATE, epoch)
    checkpoint = os.path.join(model_save_dir, ckpt_name)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': None,
        'optimizer': optimizer.state_dict(),
    }, filename=checkpoint)


def save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler):
    cuda_rng_state = 0
    if model.device == 'cuda':
        cuda_rng_state = torch.cuda.get_rng_state()

    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': cfg,
                  'random_python': random.getstate(),
                  'random_numpy': np.random.get_state(),
                  'random_pytorch': torch.get_rng_state(),
                  'random_cuda': cuda_rng_state,
                  }

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    print_log(log_path, 'Saving model at epoch %d to %s' % (epoch, model_save_dir))

    save_path = os.path.join(model_save_dir, f'ckpt_epoch_{epoch:02d}.pth')
    torch.save(save_state, save_path)


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname
