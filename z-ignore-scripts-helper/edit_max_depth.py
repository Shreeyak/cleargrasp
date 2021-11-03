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


EXT_DEPTH = '-transparent-depth-img.exr'
EXT_DEPTH_GT = '-opaque-depth-img.exr'
EXT_NORMALS = '-normals.exr'
EXT_NORMALS_RGB = '-normals-rgb.png'

parser = argparse.ArgumentParser(description='Convert folder of depth EXR images to png')
parser.add_argument('-p', '--path', required=True, help='Path to dir', metavar='path/to/dir')
args = parser.parse_args()

dir_exr = args.path
# depth_file_list = sorted(glob.glob(os.path.join(dir_exr, '*.exr')))
prefix_list = ['{:09d}'.format(i) for i in range(260, 314)]

MAX_DIST = 1.05
print('Converting EXR files to RGB files in dir: {}'.format(dir_exr))
for prefix in prefix_list:
    depth_gt_file = os.path.join(args.path, prefix + EXT_DEPTH_GT)
    depth_file = os.path.join(args.path, prefix + EXT_DEPTH)
    normals_file = os.path.join(args.path, prefix + EXT_NORMALS)
    normals_rgb_file = os.path.join(args.path, prefix + EXT_NORMALS_RGB)

    # Depth
    depth_img = api_utils.exr_loader(depth_file, ndim=1)
    depth_img[np.isnan(depth_img)] = 0.0
    depth_img[np.isinf(depth_img)] = 0.0
    depth_img[depth_img > MAX_DIST] = 0.0
    api_utils.exr_saver(os.path.join(args.path, 'depth_modified', prefix + EXT_DEPTH), depth_img, ndim=3)

    # Depth GT
    depth_gt_img = api_utils.exr_loader(depth_gt_file, ndim=1)
    depth_gt_img[np.isnan(depth_gt_img)] = 0.0
    depth_gt_img[np.isinf(depth_gt_img)] = 0.0
    depth_gt_img[depth_gt_img > MAX_DIST] = 0.0
    mask = (depth_gt_img == 0.0)
    api_utils.exr_saver(os.path.join(args.path, 'depth_modified', prefix + EXT_DEPTH_GT), depth_gt_img, ndim=3)

    # cv2.imshow('mask', mask.astype(np.uint8) * 255)
    # cv2.imshow('depth', depth_img)
    # cv2.waitKey(0)

    # Normals
    normals = api_utils.exr_loader(normals_file, ndim=3)
    normals[:, mask] = -1.0
    api_utils.exr_saver(os.path.join(args.path, 'depth_modified', prefix + EXT_NORMALS), normals, ndim=3)

    # Normals RGB
    normals_rgb = imageio.imread(normals_rgb_file)
    normals_rgb[mask, :] = 0
    imageio.imwrite(os.path.join(args.path, 'depth_modified', prefix + EXT_NORMALS_RGB), normals_rgb)

    print('Converted image {}'.format(prefix))