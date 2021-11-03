#!/usr/bin/env python3

import os
from termcolor import colored
import fnmatch
import argparse
import json
import shutil
import glob
import concurrent.futures
import time
import tqdm

import OpenEXR
import Imath
import numpy as np
from PIL import Image
from pathlib import Path
import imageio
import cv2

import torch
import torchvision
from torchvision import transforms, utils
from torch import nn
from skimage.transform import resize

from utils import exr_loader, exr_saver


depth_dir = os.path.join('../data/datasets/test/realsense-captures/source-files/depth-imgs')
ext = '*.npy'
new_ext = '.exr'

depth_files_npy = glob.glob(os.path.join(depth_dir, ext))

for depth_file in depth_files_npy:
    depth_img = np.load(depth_file)

    filename = os.path.basename(depth_file)
    filename_no_ext = os.path.splitext(depth_file)[0]
    print(filename, depth_img.shape)

    new_filename = os.path.join(filename_no_ext + new_ext)
    depth_img = np.stack((depth_img, depth_img, depth_img), axis=0)
    exr_saver(new_filename, depth_img, ndim=3)