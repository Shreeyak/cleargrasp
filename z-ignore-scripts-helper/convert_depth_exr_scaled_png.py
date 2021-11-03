"""This script will convert depth files in .exr format containing metric depth to .png files with depth
scaled by a factor of 1000 for purposes of creating point cloud animations

"""
import argparse
import os
import glob
import sys

import imageio
import numpy as np

sys.path.append('..')
from api import utils as api_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous numbered images in a dataset, move to new folder and process.')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()

    EXT_DEPTH_EXR = '.exr'
    EXT_DEPTH_PNG = '.png'
    DEPTH_SCALING = 1000

    depth_files_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_EXR)))
    if len(depth_files_list) == 0:
        raise ValueError('No files in source dir {}'.format(args.source_dir))

    print('Going to convert {} files'.format(len(depth_files_list)))
    for depth_file in depth_files_list:
        depth_img = api_utils.exr_loader(depth_file, ndim=1)
        depth_img[np.isnan(depth_img)] = 0
        depth_img[np.isinf(depth_img)] = 0
        depth_img[depth_img > 2.0] = 0

        depth_img = (depth_img * DEPTH_SCALING).astype(np.uint16)

        prefix = os.path.splitext(os.path.basename(depth_file))[0]
        depth_file_png = os.path.join(args.source_dir, prefix + EXT_DEPTH_PNG)
        imageio.imwrite(depth_file_png, depth_img)

        print('Saved file: {}'.format(os.path.basename(depth_file_png)))