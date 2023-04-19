import os
import hashlib
import requests
from tqdm import tqdm
import torch.nn as nn

import torch

__all__ = ['load_detr_weights', 'deploy_model', 'load_model', 'save_model', 'save_checkpoint', 'check_sha1', 'download']

def load_detr_weights(model, pretrain_dir, cfg):
    checkpoint = torch.load(pretrain_dir, map_location='cpu')
    model_dict = model.state_dict()

    pretrained_dict = {}
    if "dab-d-tuber-detr" in cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR:
        l = 0
    else:
        l = 1        
    for k, v in checkpoint['model'].items():
        if k.split('.')[l] == 'transformer':
            pretrained_dict.update({k: v})
            if 'united' in cfg.CONFIG.LOG.EXP_NAME and 'linear1.weight' in k and 'encoder' in k:
                pretrained_dict.update({k:v[:1024]})
            elif 'encoder' in k and "sampling_offsets" in k:
                pretrained_dict.update({k:torch.cat((v, v[:128]))})
            elif 'united' in cfg.CONFIG.LOG.EXP_NAME and 'linear2.weight' in k and 'encoder' in k:
                pretrained_dict.update({k:v[:1024]})
            elif 'cloca' in cfg.CONFIG.LOG.EXP_NAME:
                new_k = 'module.transformer2.' + ".".join(k.split('.')[2:])
                pretrained_dict.update({new_k: v})
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
            print("query_size:", query_size)
            # pretrained_dict.update({k: v[:query_size]})
    pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict} # model_dict에는 "module.query_embed.weight"라는 key가 있음
    unused_dict = {k: v for k, v in pretrained_dict.items() if not k in model_dict}
    # not_found_dict = {k: v for k, v in model_dict.items() if not k in pretrained_dict}
    # print(pretrained_dict_["module.query_embed.weight"].shape)
    print("number of detr unused model layers:", len(unused_dict.keys()))
    print("number of detr used model layers:", len(pretrained_dict_.keys()))
    # print("model_dict",[i for i in model_dict.keys()][:10])
    # print("not found layers:", not_found_dict.keys())

    model_dict.update(pretrained_dict_)
    model.load_state_dict(model_dict)
    if len(pretrained_dict_.keys())!=0:
        print("detr load pretrain success")
    else:
        print("detr load pretrain failed")


def deploy_model(model, cfg, is_tuber=True):
    """
    Deploy model to multiple GPUs for DDP training.
    """
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
        print("loading detr")
        load_detr_weights(model, cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR, cfg)
    else:
        print("detr is not loaded")

    return model


def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        print("=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
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
            print("number of unused model layers:", len(unused_dict.keys()))
            print("number of not found layers:", len(not_found_dict.keys()))
            
        else:
            pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k[7:] in model_dict}
            unused_dict = {k: v for k, v in checkpoint["model"].items() if not k[7:] in model_dict}
            not_found_dict = {k: v for k, v in model_dict.items() if not "module."+k in checkpoint["model"]}
            print("number of loaded model layers:", len(pretrained_dict.keys()))
            print("number of unused model layers:", len(unused_dict.keys()))
            print("number of not found layers:", len(not_found_dict.keys()))

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        try:
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
        except:
             print("=> loaded checkpoint '{}' (epoch 0)".format(cfg.CONFIG.MODEL.PRETRAINED_PATH, 0))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None


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
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': cfg}

    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('Saving model at epoch %d to %s' % (epoch, model_save_dir))

    save_path = os.path.join(model_save_dir, f'ckpt_epoch_{epoch}.pth')
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
