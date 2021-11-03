"""
Crops the generated sequences for ease of use in supplementary pdf

Expected format:
root_dir
├── <img_num>
│   ├── dense
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── gt
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── input
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── ours
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   └── yinda
│       ├── 000000000.jpg
│       ├── 000000001.jpg
│       ├── ...
|
├── 000000156
│   ├── dense
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
.
.
.
"""
import argparse
import concurrent.futures
import os
import glob
import itertools

import numpy as np
import open3d as o3d
import tqdm
import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a seq of images for animation of pointcloud')
    parser.add_argument('--dir_root', type=str, help='path to folder of sequences of images')
    parser.add_argument('--dir_dest', type=str, help='path to folder to save outputs')
    args = parser.parse_args()

    SUBDIR_IN = 'input'
    SUBDIR_GT = 'gt'
    SUBDIR_OURS = 'without_mask'  # 'ours'
    SUBDIR_YINDA = 'without_contact_edge'  #'yinda'
    SUBDIR_DENSE = 'without_edge_weights'  #'dense'
    dir_root = args.dir_root

    output_dir = args.dir_dest
    os.makedirs(output_dir, exist_ok=True)

    list_images = sorted(os.listdir(dir_root))
    for dir_img in list_images:
        dir_img_output = os.path.join(output_dir, dir_img)
        os.makedirs(dir_img_output, exist_ok=True)

        dir_img_input = os.path.join(dir_root, dir_img)
        f_list_in = sorted(glob.glob(os.path.join(dir_img_input, SUBDIR_IN, '*.jpg')))[:120]
        f_list_gt = sorted(glob.glob(os.path.join(dir_img_input, SUBDIR_GT, '*.jpg')))[:120]
        f_list_ours = sorted(glob.glob(os.path.join(dir_img_input, SUBDIR_OURS, '*.jpg')))[:120]
        f_list_yinda = sorted(glob.glob(os.path.join(dir_img_input, SUBDIR_YINDA, '*.jpg')))[:120]
        f_list_dense = sorted(glob.glob(os.path.join(dir_img_input, SUBDIR_DENSE, '*.jpg')))[:120]
        print('len:', len(f_list_in), len(f_list_gt), len(f_list_ours), len(f_list_yinda), len(f_list_dense))

        print('Converting image {}'.format(dir_img))
        for (f_in, f_gt, f_ours, f_yinda, f_dense) in zip(f_list_in, f_list_gt, f_list_ours, f_list_yinda, f_list_dense):
            prefix = os.path.basename(f_in)[:9]
            print('prefix:', prefix)

            pt_in = imageio.imread(f_in)
            pt_gt = imageio.imread(f_gt)
            pt_ours = imageio.imread(f_ours)
            pt_yinda = imageio.imread(f_yinda)
            pt_dense = imageio.imread(f_dense)

            _h, _w = pt_in.shape[0], pt_in.shape[1]
            _h1, _h2 = int(0.25 * _h), int(0.75 * _h)
            _w1, _w2 = int(0.25 * _w), int(0.75 * _w)

            pt_in = pt_in[_h1:_h2, _w1:_w2, :]
            pt_gt = pt_gt[_h1:_h2, _w1:_w2, :]
            pt_ours = pt_ours[_h1:_h2, _w1:_w2, :]
            pt_yinda = pt_yinda[_h1:_h2, _w1:_w2, :]
            pt_dense = pt_dense[_h1:_h2, _w1:_w2, :]

            imageio.imwrite(os.path.join(dir_img_output, '{}-in.jpg'.format(prefix)), pt_in)
            imageio.imwrite(os.path.join(dir_img_output, '{}-gt.jpg'.format(prefix)), pt_gt)
            imageio.imwrite(os.path.join(dir_img_output, '{}-without-mask.jpg'.format(prefix)), pt_ours)
            imageio.imwrite(os.path.join(dir_img_output, '{}-without-contact-edges.jpg'.format(prefix)), pt_yinda)
            imageio.imwrite(os.path.join(dir_img_output, '{}-without-edge-weights.jpg'.format(prefix)), pt_dense)
