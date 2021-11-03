"""
Script that creates gifs from sequences of jpg images, stored in folders, created from another script
Expected format:

root_dir
├── <img_num>
│   ├── dense
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── gt
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── input
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   ├── ours
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── ...
│   └── yinda
│       ├── 000000000.jpg
│       ├── 000000001.jpg
│       ├── ...
|
├── 000000156
│   ├── dense
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
.
.
.

"""
import argparse
import concurrent.futures
import glob
import io
import itertools
import os

import cv2
import tqdm
import imageio
import numpy as np
from PIL import Image


def create_individual_gifs_from_folders(dir_source, frame_duration=5):
    """
    Create a gif for each subfolder in given dir. Each subfolder should contain seq of jpg images

    Args:
        dir_source (str): Dir in which subfolders exist
        list_subdir (str): List of subfolders
        frame_duration (int): How many 10's of millis each img in seq should represent in gif

    Returns:

    """
    print('Creating gif for img: {}'.format(dir_source))
    list_subdir = []
    for root, dirs, files in os.walk(dir_source):
        for name in sorted(dirs):
            list_subdir.append(name)
        break

    for _dir in list_subdir:
        filename = os.path.join(dir_source, '{}_{}.gif'.format(os.path.basename(dir_source), _dir))
        search_str = os.path.join(dir_source, _dir, '*.jpg')
        cmd = "convert -delay {} \"{}\" -loop 0 {}".format(frame_duration, search_str, filename)
        os.system(cmd)


def create_io_gif_from_folders(dir_source, subdir_list, dir_depth=None, frame_duration=120, crop_percentage=0.3,
                               optimize=True):
    """
    Create a gif composed of a grid of images. Either a 1x2 grid of input and output or
    a 2x2 grid of gt, ours, yinda, dense.

    Args:
        dir_source (str): The root dir of 1 image, within which 5 subdirs exists containing sequences of .jpg files
        subdir_list (list): Type str. The subdirs to access to create gif. The order indicates the order of grid.
        frame_duration (int): Number of millis to display each frame
        crop_percentage (float): Value in range [0, 1], representing percentage of image to crop from each side.
        dir_depth (str): If given, For io mode, creates a 4x4 grid of in/out depth and ptcloud gif
        optimize (bool): If true, will use gifsicle to optimize output GIF files. Need to build and install gifsicle
                         manually. Please check:
                         https://github.com/kohler/gifsicle
    Returns:

    """
    if len(subdir_list) == 2:
        # input, ours
        _mode = 'io'
    elif len(subdir_list) == 4:
        # input, ours, yinda, dense
        _mode = 'compare'
    else:
        raise ValueError('Num of subdirs must be 2 or 4')

    for subdir in subdir_list:
        if not os.path.exists(os.path.join(dir_source, subdir)):
            print('[WARN]: Found dir without given subdir ({}). Skipping: {}'.format(subdir, dir_source))
            return False

    img_list = []
    for _dir in subdir_list:
        _img_list = sorted(glob.glob(os.path.join(dir_source, _dir, "*.jpg")))
        img_list.append(_img_list)

    img_list = np.array(img_list)
    _img_seq = []
    OUT_HEIGHT = 360
    OUT_WIDTH = 636
    for ii in range(img_list.shape[1]):
        _f_img_list = img_list[:, ii]
        _img_pairs = []
        label_list_io = ['Input', 'Output']
        label_list_comp = ['Ground Truth', 'Ours', 'DeepCompletion', 'DenseDepth']
        for iii, _f_img in enumerate(_f_img_list):
            _img = imageio.imread(_f_img)
            h, w = _img.shape[:2]
            _img = _img[int(crop_percentage * h):int((1-crop_percentage) * h),
                        int(crop_percentage * w):int((1-crop_percentage) * w)]
            _img = cv2.resize(_img, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)

            # Put text on the image
            if _mode == 'io':
                label_list = label_list_io
            else:
                label_list = label_list_comp
            text = label_list[iii]
            font = cv2.FONT_HERSHEY_DUPLEX  # cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1.2
            font_thickness = 2
            textsize = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            textX = int((_img.shape[1] - textsize[0]) / 2)
            textY = int(textsize[1] + 20)
            cv2.putText(_img, text, (textX, textY), font, font_scale, (0, 0, 0), font_thickness)

            _img_pairs.append(_img)

        if _mode == 'io':
            if dir_depth is None:
                _img_paired = np.hstack(_img_pairs)
            else:
                _depth_in = imageio.imread(os.path.join(dir_depth, os.path.basename(dir_source)+'-transparent-depth-rgb.png'))
                _depth_in = cv2.resize(_depth_in, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
                _depth_out = imageio.imread(os.path.join(dir_depth, os.path.basename(dir_source)+'-output-depth-rgb.png'))
                _depth_out = cv2.resize(_depth_out, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)

                _img_paired1 = np.hstack([_depth_in, _depth_out])
                _img_paired2 = np.hstack(_img_pairs)
                _img_paired = np.vstack((_img_paired1, _img_paired2))
        elif _mode == 'compare':
            _img_paired1 = np.hstack(_img_pairs[:2])
            _img_paired2 = np.hstack(_img_pairs[-2:])
            _img_paired = np.vstack((_img_paired1, _img_paired2))

        # drop every 2nd frame
        if (ii % 3) != 0:
            continue
        _img_paired = cv2.resize(_img_paired, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        _img_paired = Image.fromarray(_img_paired)

        # Saving/opening is needed for better compression and quality
        fobj = io.BytesIO()
        _img_paired.save(fobj, 'GIF')
        _img_paired = Image.open(fobj)

        _img_seq.append(_img_paired)

    # Save into a GIF file that loops forever
    filename = os.path.join(dir_source, '{}_{}_{}.gif'.format(os.path.basename(dir_source), _mode, frame_duration))
    _img_seq[0].save(filename, format='GIF', append_images=_img_seq[1:], save_all=True, duration=frame_duration, loop=0,
                     optimize=True, disposal=1)
    # Optimize GIF - requires to build and install gifsicle
    if optimize is True:
        try:
            os.system("gifsicle --batch --optimize=3 --lossy=100 --no-warnings {}".format(filename))
        except:
            print('[ERROR]: Could not optimize using gifsicle. Make sure it is manually built and installed: https://github.com/kohler/gifsicle')

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a seq of images for animation of pointcloud')
    parser.add_argument('--dir_source', type=str, help='path to folder of sequences')
    parser.add_argument('--mode', type=str, help='What to do: \
                                                 "indiv" will create 1 gif each for in, gt, ours, yinda, dense,\
                                                 "io" will create a gif of in and ours side by side,\
                                                 "compare" will create a 4x4 gif of gt, ours, yinda, dense')
    parser.add_argument('--crop_percentage', type=float, default=0.3, help='Percentage of image to crop from each side \
                                                                            to create gif from seq of images')
    parser.add_argument('--dir_depth', type=str, default=None, help='path to folder of input/output depth images')
    args = parser.parse_args()

    VALID_MODES = ['indiv', 'io', 'compare']
    if args.mode not in VALID_MODES:
        raise ValueError('Invalid mode passed: {}, must be one of: {}'.format(args.mode, VALID_MODES))

    SUBDIR_IN = 'input'
    SUBDIR_GT = 'gt'
    SUBDIR_OURS = 'ours'
    SUBDIR_YINDA = 'yinda'
    SUBDIR_DENSE = 'dense'

    print("Creating Gifs for all image sequences found in : {}".format(args.dir_source))

    # Get list of subdirs of images
    list_dir_imgs = []
    for root, dirs, files in os.walk(args.dir_source):
        for name in sorted(dirs):
            list_dir_imgs.append(os.path.join(args.dir_source, name))
        break

    # Get list of subdirs of seq type per image (in, gt, ours, yinda, densedepth) and create gif of each
    FRAME_DURATION = 5
    if args.mode == 'indiv':
        # create 1 gif each for in, gt, ours, yinda, dense
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm.tqdm(executor.map(create_individual_gifs_from_folders,
                                                  list_dir_imgs,
                                                  itertools.repeat(FRAME_DURATION)),
                                     total=len(list_dir_imgs)))
    elif args.mode == 'io':
        # create a gif of in and ours side by side
        # create_io_gif_from_folders(list_dir_imgs[0], subdir_list=[SUBDIR_IN, SUBDIR_OURS])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm.tqdm(
                executor.map(create_io_gif_from_folders,
                             list_dir_imgs,
                             itertools.repeat([SUBDIR_IN, SUBDIR_OURS]),
                             itertools.repeat(args.dir_depth)),
                total=len(list_dir_imgs)))

    elif args.mode == "compare":
        # create a gif of gt, ours, yinda, densedepth in 4x4 grid
        # create_io_gif_from_folders(list_dir_imgs[5], subdir_list=[SUBDIR_GT, SUBDIR_OURS, SUBDIR_YINDA, SUBDIR_DENSE])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm.tqdm(
                executor.map(create_io_gif_from_folders,
                             list_dir_imgs,
                             itertools.repeat([SUBDIR_GT, SUBDIR_OURS, SUBDIR_YINDA, SUBDIR_DENSE])),
                total=len(list_dir_imgs)))
