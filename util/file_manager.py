import os

import cv2
import numpy as np
import torch
from PIL import Image

from .generator import tensor2np

class FileManager:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder

        # mkdir
        for directory in ['checkpoint', 'img', 'tboard']:
            self.make_dir(directory)

    def is_dir_exist(self, dir_name:str) -> bool:
        return os.path.isdir(os.path.join(self.output_folder, dir_name))

    def make_dir(self, dir_name:str) -> str:
        os.makedirs(os.path.join(self.output_folder, dir_name), exist_ok=True)

    def get_dir(self, dir_name:str) -> str:
        # -> './output/<session_name>/dir_name'
        return os.path.join(self.output_folder, dir_name)

    def save_img_tensor(self, dir_name:str, file_name:str, img:torch.Tensor, ext='png'):
        self.save_img_numpy(dir_name, file_name, tensor2np(img), ext)

    def save_img_tensor_denorm(self, dir_name:str, file_name:str, img:torch.Tensor, norm_factor:float, ext='png'):
        """Save a tensor image with denormalization, supporting uint16 output via PIL."""
        img_np = tensor2np(img)
        # Denormalize: [0,1] -> original range
        img_np = img_np * norm_factor
        self.save_img_numpy_denorm(dir_name, file_name, img_np, norm_factor, ext)

    def save_img_numpy(self, dir_name:str, file_name:str, img:np.array, ext='png'):
        file_dir_name = os.path.join(self.get_dir(dir_name), '%s.%s'%(file_name, ext))
        if np.shape(img)[2] == 1:
            cv2.imwrite(file_dir_name, np.squeeze(img, 2))
        else:
            cv2.imwrite(file_dir_name, img)

    def save_img_numpy_denorm(self, dir_name:str, file_name:str, img:np.array, norm_factor:float, ext='png'):
        """Save a denormalized numpy image via PIL, supporting uint16."""
        file_dir_name = os.path.join(self.get_dir(dir_name), '%s.%s'%(file_name, ext))
        # Determine bit depth from norm_factor
        if norm_factor <= 255.0:
            img_np = np.clip(np.round(img), 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(np.round(img), 0, 65535).astype(np.uint16)
        # Squeeze single channel for PIL
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            # BGR -> RGB for PIL
            img_np = np.ascontiguousarray(np.flip(img_np, axis=2))
        Image.fromarray(np.ascontiguousarray(img_np)).save(file_dir_name)
    