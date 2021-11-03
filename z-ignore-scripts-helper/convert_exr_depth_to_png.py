#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import imageio

sys.path.append('..')
from api import utils as api_utils


parser = argparse.ArgumentParser(description='Convert folder of depth EXR images to png')
parser.add_argument('-p', '--path', required=True, help='Path to dir', metavar='path/to/dir')
args = parser.parse_args()

dir_exr = args.path
depth_file_list = sorted(glob.glob(os.path.join(dir_exr, '*.exr')))

COLORMAP = cv2.COLORMAP_TWILIGHT_SHIFTED
# COLORMAP = cv2.COLORMAP_JET

print('Converting EXR files to RGB files in dir: {}'.format(dir_exr))
for depth_file in depth_file_list:
    depth_img = api_utils.exr_loader(depth_file, ndim=1)
    depth_img_rgb = api_utils.depth2rgb(depth_img, min_depth=0.0, max_depth=3.0, color_mode=COLORMAP,
                                        reverse_scale=False, dynamic_scaling=True)

    depth_filename_rgb = os.path.splitext(depth_file)[0] + '.png'
    imageio.imwrite(depth_filename_rgb, depth_img_rgb)

    print('Converted image {}'.format(os.path.basename(depth_file)))
