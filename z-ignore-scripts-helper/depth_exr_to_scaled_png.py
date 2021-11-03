import os
import warnings
from termcolor import colored
import fnmatch
import numpy as np
import OpenEXR
import Imath
import shutil
import glob
import concurrent.futures
import argparse

from PIL import Image
from pathlib import Path
from scipy.misc import imsave

from torch import nn
from sklearn import preprocessing
from skimage.transform import resize

from utils import exr_loader


def main():
    '''Converts dataset of float32 depth.exr images to scaled 16-bit png images with holes

    This script takes in a dataset of depth images in a float32 .exr format.
    Then it cuts out a hole in each and converts to a scaled uint16 png image.
    These modified depth images are used as input to the depth2depth module.
    '''
    parser = argparse.ArgumentParser(
        description='Dataset Directory path')
    parser.add_argument('-p', '--depth-path', required=True,
                        help='Path to directory containing depth images', metavar='path/to/dataset')
    parser.add_argument('-l', '--height', help='The height of output image', type=int, default=288)
    parser.add_argument('-w', '--width', help='The width of output image', type=int, default=512)
    args = parser.parse_args()

    # create a directory for depth scaled png images, if it doesn't exist
    depth_imgs = os.path.join(args.depth_path, 'input-depth-scaled')

    if not os.path.isdir(depth_imgs):
        os.makedirs(depth_imgs)
        print("    Created dir:", depth_imgs)
    else:
        print("    Output Dir Already Exists:", depth_imgs)
        print("    Will overwrite files within")

    # read the exr file as np array, scale it and store as png image
    scale_value = 4000
    print('Converting depth files from exr format to a scaled uin16 png format...')
    print('Will make a portion of the img zero during conversion to test depth2depth executable')

    for root, dirs, files in os.walk(args.depth_path):
        for filename in sorted(fnmatch.filter(files, '*depth.exr')):
            name = filename[:-4] + '.png'
            np_image = exr_loader(os.path.join(args.depth_path, filename), ndim=1)
            height, width = np_image.shape

            # Create a small rectangular hole in input depth, to be filled in by depth2depth module
            h_start, h_stop = (height // 8) * 2, (height // 8) * 6
            w_start, w_stop = (width // 8) * 5, (width // 8) * 7

            # Make half the image zero for testing depth2depth
            np_image[h_start:h_stop, w_start:w_stop] = 0.0

            # Scale the depth to create the png file for depth2depth
            np_image = np_image * scale_value
            np_image = np_image.astype(np.uint16)

            # Convert to PIL
            array_buffer = np_image.tobytes()
            img = Image.new("I", np_image.T.shape)
            img.frombytes(array_buffer, 'raw', 'I;16')

            # Resize and save
            img = img.resize((args.width, args.height), Image.BILINEAR)
            img.save(os.path.join(depth_imgs, name))

    print('total ', len([name for name in os.listdir(depth_imgs) if os.path.isfile(
        os.path.join(depth_imgs, name))]), ' converted from exr to png')


if __name__ == "__main__":
    main()
