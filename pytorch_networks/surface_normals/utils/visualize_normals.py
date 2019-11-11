#!/usr/bin/env python3

"""This file is used to create RGB visualizations of the scaled PNG depth images.
"""

import sys
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import argparse
import fnmatch
import h5py

parser = argparse.ArgumentParser(description='Dataset Directory path')
parser.add_argument('-p', '--depth-path', required=True,
                    help='Path to directory containing normals images', metavar='path/to/dataset')
args = parser.parse_args()

def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).
    '''
    camera_normal_rgb = normals_to_convert + 1
    camera_normal_rgb *= 127.5
    camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb

if not os.path.isdir(args.depth_path):
    raise Exception('Directory does not exist')
    exit()

new_viz_path = os.path.join(args.depth_path, 'viz')
if not os.path.isdir(new_viz_path):
    os.makedirs(new_viz_path)

for root, dirs, files in os.walk(args.depth_path):
    for filename in sorted(fnmatch.filter(files, '*.h5')):
        #im = Image.open(os.path.join(args.depth_path, filename))
        #im = np.array(im)
        f = h5py.File(os.path.join(args.depth_path, filename), 'r')
        dset = f['result']
        im = dset[()]
        im = im.transpose((1, 2, 0))
        im = normal_to_rgb(im)
        print('im', im.shape)

        fig = plt.figure()
        ax0 = plt.subplot(111)
        ax0.imshow(im)
        ax0.set_title('Surface Normal Image')  # subplot 211 title
        # plt.show()

        fig.savefig(os.path.join(args.depth_path, 'viz', os.path.splitext(filename)[0] + '.png'))
        plt.close('all')
    break

