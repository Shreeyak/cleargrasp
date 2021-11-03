"""Some images are of diff resolution from the rest. Resize a folder full of images of diff file formats to a new resolution.
"""
import argparse
import glob
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.append('..')
import api.utils as api_utils

parser = argparse.ArgumentParser(
    description='Resize a given depth/color image')

parser.add_argument('--path', required=True, help='Path to dir', metavar='path/to/dir')
parser.add_argument('--height', default=480, type=int,
                    help='Vertical FOV of camera in radians (Field of View along the height of image)')
parser.add_argument('--width', default=848, type=int,
                    help='Horizontal FOV of camera in radians (Field of View along the width of image)')
args = parser.parse_args()

filenames = sorted(glob.glob(os.path.join(args.path, '*')))

for filename in filenames:
    prefix, ext = os.path.splitext(filename)

    if ext == '.exr':
        image = api_utils.exr_loader(filename, ndim=1)
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        api_utils.exr_saver(filename, image, ndim=3)
    elif ext == '.jpg':
        image = imageio.imread(filename)
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
        imageio.imwrite(filename, image, quality=100)
    elif ext == '.png':
        image = imageio.imread(filename)
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        imageio.imwrite(filename, image)
    else:
        print('Unknown file format, skipping: {}'.format(filename))

    print('Resized {}'.format(filename))
