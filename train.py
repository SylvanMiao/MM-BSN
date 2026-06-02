"""
MM-BSN CONFOCAL — Training Script
==================================
Train MM-BSN on confocal microscopy images for self-supervised denoising.

Data for training is loaded via DataDeal/CONFOCAL.py.
Edit that file to set your dataset path BEFORE running this script.


│  QUICK START                                                     
                                                                  
  # 1) Train from scratch (single GPU)                            
  python train.py -c ./config/CONFOCAL -g 0 \\                   
      -sd ./output/Confocal                                       
                                                                  
  # 2) Resume from checkpoint                                     
  python train.py -c ./config/CONFOCAL -g 0 -r \\                
      -p ./output/Confocal/checkpoint/CONFOCAL_MMBSN_050.pth \\  
      -sd ./output/Confocal                                       
                                                                  
  # 3) Multi-GPU training                                         
  python train.py -c ./config/CONFOCAL -g 0,1,2,3 \\             
      -sd ./output/Confocal                                       



  ARGUMENTS                                                       
                                                                  
  -c, --config         Config file (without .yaml extension).     
                       Default: config/CONFOCAL                   
  -g, --gpu            GPU ID(s). Use 'cpu' for CPU mode.         
                       Multi-GPU: -g 0,1,2,3                      
  -r, --resume         Resume training from checkpoint.           
  -p, --pretrained     Checkpoint path for resuming.              
  -sd, --ckpt_save_folder  Output directory (checkpoints/logs).   
                       Default: output/Confocal                   
  -rd, --data_root_dir Root dataset path (default: ./dataset).    
                       The actual confocal data path is           
                       hardcoded in DataDeal/CONFOCAL.py.         
  -t, --thread         DataLoader workers (default: 4).           



  BEFORE TRAINING                                                 
                                                                  
  1. Edit DataDeal/CONFOCAL.py → _scan() → set dataset_path to   
     your confocal image folder:                                  
       dataset_path = '../../../Dataset/T3/your_confocal_folder'  
                                                                  
  2. Review config/CONFOCAL.yaml for training hyperparameters:    
     - batch_size, max_epoch, init_lr, scheduler                  
     - crop_size (must be divisible by pd_a=4)                    
     - mask_type (e.g. 'o_a45', 'o_fsz')                          
     - norm_factor (65535.0 for 16-bit, 255.0 for 8-bit)          
                                                                  
  3. Ensure the dataset_path directory exists and contains        
     .png / .tif / .tiff images.                                  



  OUTPUT STRUCTURE                                                
                                                                  
  output/Confocal/                                                
  ├── checkpoint/           Model checkpoints                     
  │   └── CONFOCAL_MMBSN_001.pth, ..._002.pth, ...               
  ├── img/                  Validation denoising results          
  │   └── val_001/                                               
  └── tboard/               TensorBoard event logs                

"""

import os
import argparse
import math
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.model_need_tools import set_module, set_optimizer, load_checkpoint, summary, set_status,\
    set_denoiser, test_dataloader_process, print_loss, warmup, _adjust_lr, _run_step
from util.config_parse import ConfigParser
from util.file_manager import FileManager
from util.logger import Logger
from util.loss import Loss
from util.generator import human_format
from DataDeal.CONFOCAL import Confocal

# Registry mapping dataset name strings to classes
_DATASET_REGISTRY = {
    'Confocal': Confocal,
}

def _get_dataset_class(name: str):
    """Look up a dataset class by name string."""
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name]


def _build_dataloader(dataset_cfg, data_root_dir, batch_size, shuffle, num_workers, drop_last=False, default_name='Confocal'):
    """Build a DataLoader from dataset config. Supports both new (keyed by dataset name)
    and old (single dataset dict) formats."""
    dataset_dict = dataset_cfg.get('dataset', None)
    if dataset_dict is None or isinstance(dataset_dict, str):
        # New format: training.dataset = 'Confocal', training.dataset_args = {...}
        dataset_name = dataset_dict if isinstance(dataset_dict, str) else default_name
        DatasetClass = _get_dataset_class(dataset_name)
        args = dict(dataset_cfg.get('dataset_args', {}))
    elif isinstance(dataset_dict, dict):
        # Old format: training.dataset = {'dataset': 'preped_RN_data'}, training.dataset_args = {...}
        dataset_name = list(dataset_dict.keys())[0]
        DatasetClass = _get_dataset_class(dataset_name)
        args = dict(dataset_cfg.get('dataset_args', {}))
    else:
        raise ValueError(f"Unsupported dataset config format: {type(dataset_dict)} = {dataset_dict}")

    # Build dataset_path: prefer explicit args, fall back to data_root_dir convention
    dataset_path = args.pop('dataset_path', None)
    if dataset_path is None:
        dataset_path = os.path.join(data_root_dir, 'Confocal')

    dataset = DatasetClass(**args, dataset_path=dataset_path)
    dataloader = {}
    dataloader['dataset'] = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False, drop_last=drop_last)
    return dataloader



def train():
    module = set_module(cfg)
    # training dataset loader
    train_dataloader = _build_dataloader(
        train_cfg, cfg['data_root_dir'],
        batch_size=train_cfg['batch_size'], shuffle=True,
        num_workers=cfg['thread'], drop_last=True,
        default_name='Confocal')

    # validation dataset loader
    val_dataloader = _build_dataloader(
        val_cfg, cfg['data_root_dir'],
        batch_size=1, shuffle=False,
        num_workers=cfg['thread'], drop_last=False,
        default_name='Confocal')
    # other configuration
    max_epoch = train_cfg['max_epoch']
    epoch = start_epoch = 1
    max_len = train_dataloader['dataset'].dataset.__len__() # base number of iteration works for dataset named 'dataset'
    max_iter = math.ceil(max_len / train_cfg['batch_size'])

    loss = Loss(train_cfg['loss'], train_cfg['tmp_info'])
    loss_dict = {'count': 0}
    tmp_info = {}
    loss_log = []

    # set optimizer
    optimizer = set_optimizer(module, train_cfg)

    for opt in optimizer.values():
        opt.zero_grad(set_to_none=True)

    # resume
    if cfg["resume"]:
        # load last checkpoint
        load_checkpoint(module, cfg, checkpoint_path=cfg['pretrained'])
        epoch = int(cfg['pretrained'].split('/')[-1].split('.')[0].split('_')[-1])+1

        # logger initialization
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''), log_file_option='a')
    else:
        # logger initialization
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''), log_file_option='w')

    # tensorboard
    tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    tboard = SummaryWriter(log_dir=file_manager.get_dir('tboard/%s'%tboard_time))

    # device setting
    if cfg['gpu'] != 'cpu':
        # model to GPU
        model = {key: nn.DataParallel(module[key]).cuda() for key in module}
        # optimizer to GPU
        for optim in optimizer.values():
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        model = {key: nn.DataParallel(module[key]).cpu() for key in module}

    # start message
    logger.info(summary(module, human_format))
    logger.start((epoch-1, 0))
    logger.highlight(logger.get_start_msg())

    if epoch == 1 and train_cfg['warmup']:
        warmup(model, loss, train_dataloader, optimizer, logger, max_iter, epoch, max_epoch, loss_dict, loss_log,tmp_info,tboard, cfg)

    # training
    for epoch in range(epoch, max_epoch + 1):
        status = set_status('epoch %03d/%03d' % (epoch, max_epoch))
        # make dataloader iterable.
        train_dataloader_iter = {}
        for key in train_dataloader:
            train_dataloader_iter[key] = iter(train_dataloader[key])

        # model training mode
        for key in model:
            model[key].train()

        for iter_id in range(1, max_iter+1):
            _run_step(train_dataloader_iter, model, optimizer, loss,epoch, iter_id, max_iter, max_epoch, loss_dict, cfg)
            _adjust_lr(optimizer, iter_id, epoch, max_iter, train_cfg)

            if (iter_id % cfg['log']['interval_iter'] == 0 and iter_id != 0) or (iter_id == max_iter):
                print_loss(optimizer, logger, loss_dict, loss_log, tmp_info, status, tboard, iter_id, max_iter, epoch)

            # print progress
            logger.print_prog_msg((epoch - 1, iter_id - 1))
        # save checkpoint
        ckpt_save_folder = cfg['ckpt_save_folder']
        if not os.path.exists(ckpt_save_folder):
            os.makedirs(ckpt_save_folder)
        checkpoint_name = cfg['config'].split('/')[-1] +'_'+ cfg['model']['kwargs']['type'] + '_%03d'%epoch + '.pth'
        if epoch >= ckpt_cfg['start_epoch']:
            if (epoch - ckpt_cfg['start_epoch']) % ckpt_cfg['interval_epoch'] == 0:
                torch.save({'epoch': epoch,
                            'model_weight': {key: model[key].module.state_dict() for key in model}},
                           os.path.join(ckpt_save_folder, 'checkpoint',checkpoint_name))

        # validation

        if val_cfg['val']:
            if epoch >= val_cfg['start_epoch']:
                if (epoch - val_cfg['start_epoch']) % val_cfg['interval_epoch'] == 0:
                    for key in model:
                        model[key].eval()
                    set_status('val %03d' % epoch)
                    checkpoint_path = os.path.join(ckpt_save_folder, 'checkpoint', checkpoint_name)
                    denoiser = set_denoiser(checkpoint_path, cfg)

                    # make directories for image saving
                    img_save_path = 'img/val_%03d' % epoch

                    file_manager.make_dir(img_save_path)

                    # count psnr/ssim and save denoised validation image
                    psnr, ssim = test_dataloader_process(    denoiser=denoiser,
                                                              dataloader=val_dataloader['dataset'],
                                                             file_manager=file_manager,
                                                              cfg=cfg,
                                                              add_con=0. if not 'add_con' in val_cfg else
                                                              val_cfg['add_con'],
                                                              floor=False if not 'floor' in val_cfg else
                                                              val_cfg['floor'],
                                                              img_save_path=img_save_path,
                                                              img_save=val_cfg['save_image'],
                                                             logger=logger,
                                                             status=status,
                                                             norm_factor=val_cfg.get('norm_factor', 1.0))

    logger.highlight(logger.get_finish_msg())

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c',  '--config',            default='config/CONFOCAL',  type=str)
    args.add_argument('-g',  '--gpu',               default='0',  type=str)
    args.add_argument('-r',  '--resume',            default=False)
    args.add_argument('-p',  '--pretrained',        default=None,  type=str)
    args.add_argument('-t',  '--thread',            default=4,     type=int)
    args.add_argument('-sd', '--ckpt_save_folder',  default='output/Confocal', type=str)
    args.add_argument('-rd', '--data_root_dir',     default='./dataset', type=str)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    train_cfg = cfg['training']
    val_cfg = cfg['validation']
    ckpt_cfg = cfg['checkpoint']
    status_len = 13
    file_manager = FileManager(cfg['ckpt_save_folder'])

    train()