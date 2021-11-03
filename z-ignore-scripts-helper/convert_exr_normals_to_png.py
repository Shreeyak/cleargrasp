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

SEARCH_STR = '*-normals.exr'
normals_file_list = sorted(glob.glob(os.path.join(dir_exr, SEARCH_STR)))

print('Converting EXR files to RGB files in dir: {}'.format(dir_exr))
for normals_file in normals_file_list:
    normals_img = api_utils.exr_loader(normals_file, ndim=3)
    normals_img = normals_img.transpose((1, 2, 0))

    normals_img_rgb = api_utils.normal_to_rgb(normals_img, output_dtype='uint8')

    normals_filename_rgb = os.path.splitext(normals_file)[0] + '.png'
    imageio.imwrite(normals_filename_rgb, normals_img_rgb)

    print('Converted image {}'.format(os.path.basename(normals_file)))
