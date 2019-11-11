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

parser = argparse.ArgumentParser(description='Dataset Directory path')
parser.add_argument('-p', '--depth-path', required=True,
                    help='Path to directory containing depth images', metavar='path/to/dataset')
args = parser.parse_args()

if not os.path.isdir(args.depth_path):
    raise Exception('Directory does not exist')
    exit()

new_viz_path = os.path.join(args.depth_path, 'viz')
if not os.path.isdir(new_viz_path):
    os.makedirs(new_viz_path)

for root, dirs, files in os.walk(args.depth_path):
    for filename in sorted(fnmatch.filter(files, '*-depth.png')):
        im = Image.open(os.path.join(args.depth_path, filename))
        im = np.array(im)

        fig = plt.figure()
        ax0 = plt.subplot(111)
        ax0.imshow(im)
        ax0.set_title('Depth Image')  # subplot 211 title
        # plt.show()

        fig.savefig(os.path.join(args.depth_path, 'viz', filename))
        plt.close('all')
    break

