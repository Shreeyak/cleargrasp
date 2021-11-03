"""Script that loads in a seq of RGBD files for animation of input/output ptclouds.
Expects RGB images in .jpg format in a folder called 'color' and Depth images in .png format in folder called 'depth'.
Depth images will contain depth scaled by some factor.
Path should also contain a 'camera_intrinsic.json' file. This file will contain the depth scaling factor.
"""
import argparse
import time
import os
import glob
import random
import sys

import cv2
import numpy as np
import open3d as o3d
import tqdm
import imageio

sys.path.append('..')
from api import utils as api_utils

# The various subfolders into which the synthetic data is to be organized into.
# These folders will be created and the files with given postfixes will be moved into them.
SUBFOLDER_MAP_SYNTHETIC = {
    'rgb-files': {
        'postfix': '-rgb.jpg',
        'folder-name': 'rgb-imgs'
    },
    'camera-normals-rgb': {
        'postfix': '-cameraNormals.png',
        'folder-name': 'camera-normals/rgb-visualizations'
    },
    'outlines-rgb': {
        'postfix': '-outlineSegmentationRgb.png',
        'folder-name': 'outlines/rgb-visualizations'
    },
    'depth-files-rectified': {
        'postfix': '-depth-rectified.exr',
        'folder-name': 'depth-imgs-rectified'
    },
    'segmentation-masks': {
        'postfix': '-segmentation-mask.png',
        'folder-name': 'segmentation-masks'
    }
}

if __name__ == "__main__":

    output_dir = 'data/syn_samples'
    if not os.path.isdir(output_dir):
        print('Creating dir to store results: {}'.format(output_dir))
        os.makedirs(output_dir)
    else:
        print("[WARN]: Folder {} already exists! Overwriting".format(output_dir))

    path1 = '/media/shrek/ Eolian/datasets-transparent/paper_dataset_synthetic'

    display_width = 512
    display_height = 288
    counter = 90
    for ii in range(10):
        r_prefix = random.randint(0, 90)
        for _dir1 in os.listdir(path1):
            for _dir in os.listdir(os.path.join(path1, _dir1)):
                try:
                    i_type = 'rgb-files'
                    rgb_img = imageio.imread(os.path.join(path1, _dir1, _dir, SUBFOLDER_MAP_SYNTHETIC[i_type]['folder-name'], '{:09d}'.format(r_prefix)+SUBFOLDER_MAP_SYNTHETIC[i_type]['postfix']))
                    rgb_img = cv2.resize(rgb_img, (display_width, display_height), interpolation=cv2.INTER_CUBIC)

                    i_type = 'depth-files-rectified'
                    depth_img = api_utils.exr_loader(os.path.join(path1, _dir1, _dir, SUBFOLDER_MAP_SYNTHETIC[i_type]['folder-name'], '{:09d}'.format(r_prefix) + SUBFOLDER_MAP_SYNTHETIC[i_type]['postfix']), ndim=1)
                    depth_img = api_utils.depth2rgb(depth_img, max_depth=2.5, dynamic_scaling=True)
                    depth_img = cv2.resize(depth_img, (display_width, display_height), interpolation=cv2.INTER_CUBIC)

                    i_type = 'camera-normals-rgb'
                    normals_img = imageio.imread(os.path.join(path1, _dir1, _dir, SUBFOLDER_MAP_SYNTHETIC[i_type]['folder-name'], '{:09d}'.format(r_prefix) + SUBFOLDER_MAP_SYNTHETIC[i_type]['postfix']))
                    normals_img = cv2.resize(normals_img, (display_width, display_height), interpolation=cv2.INTER_CUBIC)

                    i_type = 'outlines-rgb'
                    outlines_img = imageio.imread(os.path.join(path1, _dir1, _dir, SUBFOLDER_MAP_SYNTHETIC[i_type]['folder-name'], '{:09d}'.format(r_prefix) + SUBFOLDER_MAP_SYNTHETIC[i_type]['postfix']))
                    outlines_img = cv2.resize(outlines_img, (display_width, display_height), interpolation=cv2.INTER_CUBIC)

                    i_type = 'segmentation-masks'
                    mask_img = imageio.imread(os.path.join(path1, _dir1, _dir, SUBFOLDER_MAP_SYNTHETIC[i_type]['folder-name'], '{:09d}'.format(r_prefix) + SUBFOLDER_MAP_SYNTHETIC[i_type]['postfix']))
                    mask_img = cv2.resize(mask_img, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
                    if len(mask_img.shape) == 2:
                        mask_img = np.stack([mask_img, mask_img, mask_img], axis=2)

                    img = np.hstack([rgb_img, depth_img, normals_img, outlines_img, mask_img])
                    # print(img.shape)
                    imageio.imwrite(os.path.join(output_dir, '{:09d}-syn-sample.png'.format(counter)), img)
                    counter += 1
                except OSError as e:
                    pass


    # # Get list of indexes
    # list_color_imgs = sorted(glob.glob(os.path.join(args.path, COLOR_SUBDIR, '*')))
    # list_depth_imgs = sorted(glob.glob(os.path.join(args.path, DEPTH_SUBDIR, '*')))
    # if len(list_color_imgs) != len(list_depth_imgs):
    #     raise ValueError('Num of color imgs {} and depth imgs {} not equal.'.format(len(list_color_imgs),
    #                                                                                 len(list_depth_imgs)))
