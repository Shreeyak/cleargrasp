import argparse
import glob
import os
import sys

import numpy as np

sys.path.append('..')
import api.utils as api_utils

parser = argparse.ArgumentParser(
    description='Multiply depth image by given scale and save')

parser.add_argument('--path', required=True, help='Path to dir', metavar='path/to/file')
parser.add_argument('--scale', default=0.1, type=float,
                    help='Multiply depth by this scale')
args = parser.parse_args()

DEPTH_EXT = '.exr'

depth_file_list = sorted(glob.glob(os.path.join(args.path, '*'+DEPTH_EXT)))

for img_path in depth_file_list:
    print('Re-scaling img: {}'.format(img_path))
    image = api_utils.exr_loader(img_path, ndim=1)
    image *= args.scale
    api_utils.exr_saver(img_path, image, ndim=3)
