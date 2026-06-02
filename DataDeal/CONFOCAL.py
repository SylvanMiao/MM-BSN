import os

import numpy as np
import torch
from PIL import Image

from .DatasetBase import RealDataSet


class Confocal(RealDataSet):
    '''
    Confocal dataset class for uint16 single-channel images.
    All images are stored in a single folder.
    Supports mixed 8-bit and 16-bit images via per-image auto-detection.
    Dataset path is hardcoded in _scan() — no need to pass dataset_path.
    '''
    def __init__(self, *args, **kwargs):
        # Always override dataset_path — the actual data path is hardcoded in _scan().
        # This bypasses RealDataSet's directory check with a dummy path that always exists.
        kwargs['dataset_path'] = '.'
        super().__init__(*args, **kwargs)

    def _scan(self):
        # Hardcoded dataset path — change this to your confocal data folder
        dataset_path = '../../../Dataset/T3/all_crop_pix512_pix1024_300pics'
        assert os.path.exists(dataset_path), 'There is no dataset %s' % dataset_path

        # Scan all PNG/TIF files in the dataset directory
        for file_name in sorted(os.listdir(dataset_path)):
            if file_name.lower().endswith(('.png', '.tif', '.tiff')):
                self.img_paths.append(os.path.join(dataset_path, file_name))

        assert len(self.img_paths) > 0, 'No images found in %s' % dataset_path
        print('Found %d confocal images in %s' % (len(self.img_paths), dataset_path))

    def _load_data(self, data_idx):
        img_path = self.img_paths[data_idx]

        # Use PIL to preserve original bit depth
        img = Image.open(img_path)
        img_np = np.array(img)

        # Force single channel: strip extra dims, keep grayscale as-is
        if img_np.ndim > 2:
            img_np = img_np[:, :, 0]
        assert img_np.ndim == 2, \
            "unexpected image shape %s for %s" % (img_np.shape, img_path)

        # Auto-detect bit depth: uint8 (max<=255) or uint16 (max>255)
        norm_factor = 255.0 if img_np.max() <= 255 else 65535.0

        # Normalize to [0, 1]
        img_np = np.expand_dims(img_np.astype(np.float32), axis=0)
        img_np = img_np / norm_factor
        noisy_img = torch.from_numpy(np.ascontiguousarray(img_np))

        return {'real_noisy': noisy_img, 'norm_factor': norm_factor}
