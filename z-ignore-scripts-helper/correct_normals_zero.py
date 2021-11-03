import argparse
import glob
import os
import sys

import numpy as np

sys.path.append('..')
import api.utils as api_utils

parser = argparse.ArgumentParser(
    description='Create mask of invalid normals')

parser.add_argument('--path', required=True, help='Path to dir', metavar='path/to/file')
args = parser.parse_args()

NORMALS_EXT = '-normals.exr'

normals_file_list = sorted(glob.glob(os.path.join(args.path, '*'+NORMALS_EXT)))

for img_path in normals_file_list:
    print('Correcting img: {}'.format(img_path))

    image = api_utils.exr_loader(img_path, ndim=3)
    mask = np.all(image==-1, axis=0)
    # image[:, mask] = (-1.0, -1.0, -1.0)
    print('mask:', np.sum(mask), mask.shape, '\n', mask)
    print('img:', image[:, mask].shape, '\n', image[:, mask])
    # api_utils.exr_saver(os.path.join(args.path, 'data_norm', os.path.basename(img_path)), image, ndim=3)
