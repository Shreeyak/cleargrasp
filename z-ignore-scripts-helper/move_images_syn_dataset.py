#!/usr/bin/env python3

import argparse
import concurrent.futures
import fnmatch
import glob
import itertools
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
import sys
from termcolor import colored

# The various subfolders into which the synthetic data is to be organized into.
# These folders will be created and the files with given postfixes will be moved into them.
SUBFOLDER_MAP_SYNTHETIC = {
    'rgb-files': {
        'postfix': '-rgb.jpg',
        'folder-name': 'rgb-imgs'
    },
    'depth-files': {
        'postfix': '-depth.exr',
        'folder-name': 'depth-imgs'
    },
    'json-files': {
        'postfix': '-masks.json',
        'folder-name': 'json-files'
    },
    'world-normals': {
        'postfix': '-normals.exr',
        'folder-name': 'world-normals'
    },
    'variant-masks': {
        'postfix': '-variantMasks.exr',
        'folder-name': 'variant-masks'
    },
    'component-masks': {
        'postfix': '-componentMasks.exr',
        'folder-name': 'component-masks'
    },
    'camera-normals': {
        'postfix': '-cameraNormals.exr',
        'folder-name': 'camera-normals'
    },
    'camera-normals-rgb': {
        'postfix': '-cameraNormals.png',
        'folder-name': 'camera-normals/rgb-visualizations'
    },
    'outlines': {
        'postfix': '-outlineSegmentation.png',
        'folder-name': 'outlines'
    },
    'outlines-rgb': {
        'postfix': '-outlineSegmentationRgb.png',
        'folder-name': 'outlines/rgb-visualizations'
    },
    'depth-files-rectified': {
        'postfix': '-depth-rectified.exr',
        'folder-name': 'depth-imgs-rectified'
    },
    'segmentation-masks': {
        'postfix': '-segmentation-mask.png',
        'folder-name': 'segmentation-masks'
    }
}


def move_to_subfolders(dest_dir, source_dir, index, dest_index=None):
    '''Move each file type to it's own subfolder.
    It will create a folder for each file type. The file type is determined from it's postfix.
    The file types and their corresponding directory are defined in the SUBFOLDER_MAP dict

    Args:
        dest_dir (str): Path to new dataset.
        source_dir (str): Path to old dataset from which to move.
        index (int): The prefix num which is to be moved.

    Returns:
        None
    '''

    count_moved = 0
    for filetype in SUBFOLDER_MAP_SYNTHETIC:
        file_postfix = SUBFOLDER_MAP_SYNTHETIC[filetype]['postfix']
        subfolder = SUBFOLDER_MAP_SYNTHETIC[filetype]['folder-name']

        filename = os.path.join(source_dir, subfolder, '{:09d}'.format(index) + file_postfix)
        if os.path.isfile(filename):
            if dest_index is None:
                dest_index = index
            shutil.move(filename, os.path.join(dest_dir, subfolder, '{:09d}'.format(dest_index)+file_postfix))
            count_moved += 1

    print("\tMoved {} to {}".format(index, dest_index))
    if count_moved > 1:
        count_moved = 1
    return count_moved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous numbered images in a dataset, move to separate folders and process.')

    parser.add_argument('-s', required=True, help='Path to source dataset', metavar='path/to/dataset')
    parser.add_argument('-d', required=True, help='Path to dest dataset', metavar='path/to/dataset')
    parser.add_argument('-n', default=0, type=int, help='Numbering in dest will strt from this')
    args = parser.parse_args()

    if not os.path.isdir(args.s):
        print(colored('ERROR: Did not find {}. Please pass correct path to dataset'.format(args.s), 'red'))
        exit()
    if not os.path.isdir(args.d):
        print(colored('ERROR: Did not find {}. Please pass correct path to dataset'.format(args.d), 'red'))
        exit()

    source_dir = args.s
    dest_dir = args.d
    for filetype in SUBFOLDER_MAP_SYNTHETIC:
        subfolder_path = os.path.join(dest_dir, SUBFOLDER_MAP_SYNTHETIC[filetype]['folder-name'])

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path)
            print("\tCreated dir:", subfolder_path)

    data = 0

    while True:
        index = input("Enter range or prefix of files to move: ")
        if index == 'q':
            print('Quitting.')
            exit()
        elif index == 'c':
            start_num = int(input("Enter starting prefix: "))
            end_num = int(input("Enter end prefix: "))
            for ii in range(start_num, end_num):
                move_to_subfolders(dest_dir, source_dir, ii)
        elif index == 'a':
            start_num = int(input("Enter starting prefix: "))
            end_num = int(input("Enter end prefix: "))
            count_moved = 0
            for ii in range(start_num, end_num):
                count_moved += move_to_subfolders(dest_dir, source_dir, ii, ii + args.n)
        else:
            index = int(index)
            move_to_subfolders(dest_dir, source_dir, index)
