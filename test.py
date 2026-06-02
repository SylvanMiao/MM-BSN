"""
MM-BSN CONFOCAL — Inference Script
===================================
Denoise confocal microscopy images (8-bit or 16-bit grayscale)
with a trained MM-BSN model.

Image loading replicates SC-BSN's PIL-based approach:
  - Auto-detects bit depth (8-bit / 16-bit)
  - Auto-converts color images to grayscale single-channel
  - Normalizes to [0, 1] using norm_factor from config


  QUICK START                                                    
                                                                 
  # 1) Batch denoise all images in a folder (PRIMARY USE CASE)   
  python test.py -c ./config/CONFOCAL -g 0 \\                   
      --pretrained ./output/Confocal/checkpoint/CONFOCAL_MMBSN_100.pth \\
      --test_dir ./your_noisy_images/ \\                         
      --save_folder ./output/denoised_results                    
                                                                 
 # 2) Evaluate on CONFOCAL dataset (with PSNR/SSIM metrics)    
 python test.py -c ./config/CONFOCAL -g 0 \\                    
     --pretrained ./output/Confocal/checkpoint/CONFOCAL_MMBSN_100.pth



  ARGUMENTS                                                      
                                                                 
  -c, --config       Config file (without .yaml extension).      
                     Default: config/CONFOCAL                    
  -g, --gpu          GPU ID. Use 'cpu' for CPU inference.        
  --pretrained       Path to trained checkpoint (.pth).          
                     MUST be a CONFOCAL-trained model.           
  --test_dir         Directory of noisy images to denoise.       
                     When set → batch denoise all images inside. 
                     When NOT set → CONFOCAL dataset evaluation. 
  --save_folder      Output directory for denoised images.       
  --thread           DataLoader workers (default: 4).            
  -rd, --data_root_dir  Dataset root (used for evaluation mode). 


"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from util.config_parse import ConfigParser
from util.file_manager import FileManager
from util.logger import Logger
from util.model_need_tools import set_denoiser, test_dataloader_process, set_status
from DataDeal.CONFOCAL import Confocal


# ======================================================================
#  Image I/O — replicating SC-BSN's PIL-based approach
# ======================================================================

def _load_image_pil(image_path, norm_factor=None):
    """
    Load image via PIL (preserves bit depth, auto grayscale conversion).

    Returns:
        tensor:   (1, C, H, W) float32
        max_val:  float  65535.0 for 16-bit, 255.0 for 8-bit
    """
    img = Image.open(image_path)

    if img.mode in ('I;16', 'I;16B', 'I;16L'):
        arr = np.array(img, dtype=np.uint16).astype(np.float32)
        native_max = 65535.0
    else:
        arr = np.array(img.convert('L'), dtype=np.uint8).astype(np.float32)
        native_max = 255.0

    if arr.ndim == 2:
        arr = arr[:, :, None]  # (H, W) -> (H, W, 1)

    if norm_factor is not None:
        arr = arr / norm_factor  # normalize to [0, 1]

    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).float()
    return tensor, native_max


def _save_image_pil(arr, path, max_val):
    """
    Save a numpy array as image. Inverse of _load_image_pil.
    Args:
        arr:     (H, W) or (H, W, C) float32 in [0, 1]
        max_val: 65535.0 or 255.0
    """
    arr = np.clip(arr * max_val, 0, max_val).round()

    if max_val > 255:
        Image.fromarray(arr.astype(np.uint16)).save(path)
    else:
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        Image.fromarray(arr.astype(np.uint8)).save(path)


# ======================================================================
#  Single-image inference
# ======================================================================

def test_img(denoiser, image_path, save_dir, norm_factor, add_con, floor, gpu):
    """Inference a single image with PIL-based loading."""
    print(image_path)

    noisy, max_val = _load_image_pil(image_path, norm_factor=norm_factor)

    if gpu != 'cpu':
        noisy = noisy.cuda()

    denoised = denoiser(noisy)
    denoised += add_con
    if floor:
        denoised = torch.floor(denoised)

    denoised_np = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    if denoised_np.shape[2] == 1:
        denoised_np = denoised_np[:, :, 0]

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, name + '_DN.png')
    _save_image_pil(denoised_np, out_path, max_val)
    print('saved : %s' % out_path)


# ======================================================================
#  Test orchestration
# ======================================================================

@torch.no_grad()
def test():
    test_cfg = cfg['test']
    norm_factor = test_cfg.get('norm_factor', 1.0)
    add_con = test_cfg.get('add_con', 0.0)
    floor = test_cfg.get('floor', False)
    gpu = cfg['gpu']

    # ---- Directory inference (primary use case) ---- #
    if cfg['test_dir'] is not None:
        os.makedirs(cfg['save_folder'], exist_ok=True)
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        for f in sorted(os.listdir(cfg['test_dir'])):
            if f.lower().endswith(exts):
                test_img(denoiser, os.path.join(cfg['test_dir'], f),
                         cfg['save_folder'], norm_factor, add_con, floor, gpu)
        return

    # ---- CONFOCAL dataset evaluation (with PSNR/SSIM) ---- #
    file_manager = FileManager(cfg['save_folder'])
    img_save_path = 'img/test_Confocal'

    test_args = test_cfg.get('dataset_args', {})
    test_dataset = Confocal(**test_args)
    dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                            num_workers=cfg['thread'], pin_memory=False)

    logger = Logger()
    logger.highlight(logger.get_start_msg())
    status = set_status('test')

    psnr, ssim = test_dataloader_process(
        denoiser=denoiser,
        file_manager=file_manager,
        cfg=cfg,
        dataloader=dataloader,
        add_con=add_con,
        floor=floor,
        img_save_path=img_save_path,
        img_save=test_cfg.get('save_image', True),
        logger=logger,
        status=status,
        norm_factor=norm_factor)

    if psnr is not None and ssim is not None:
        result_file = os.path.join(file_manager.get_dir(img_save_path),
                                   '_psnr-%.2f_ssim-%.3f.result' % (psnr, ssim))
        with open(result_file, 'w') as f:
            f.write('PSNR: %f\nSSIM: %f' % (psnr, ssim))


# ======================================================================
#  Entry point
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MM-BSN CONFOCAL Inference')
    parser.add_argument('-c', '--config',       default='config/CONFOCAL', type=str)
    parser.add_argument('-g', '--gpu',          default='0', type=str)
    parser.add_argument('--save_folder',        default='output/Confocal_test', type=str)
    parser.add_argument('--pretrained',         default='./ckpt/MMBSN_SIDD_o_a45.pth', type=str)
    parser.add_argument('--thread',             default=4, type=int)
    parser.add_argument('--test_dir',           default=None, type=str,
                        help='Directory of noisy images for batch inference')
    parser.add_argument('-rd', '--data_root_dir', default='./dataset', type=str)

    args = parser.parse_args()
    assert args.config is not None, 'config file path is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    if not os.path.isdir(cfg['save_folder']):
        os.makedirs(cfg['save_folder'])

    print('Checkpoint: %s' % cfg['pretrained'])
    denoiser = set_denoiser(checkpoint_path=cfg['pretrained'], cfg=cfg)

    test()
