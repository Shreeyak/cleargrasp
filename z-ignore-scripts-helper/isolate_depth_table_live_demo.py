"""Script to remove depth of all items not on table.
"""
import argparse
import time
import os
import json
import glob

import cv2
import numpy as np
import open3d as o3d
import tqdm
import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a seq of images for animation of pointcloud')
    parser.add_argument('-s', '--source_dir', type=str, help='path to pointcloud')
    parser.add_argument('-k', '--kernel', type=int, default=5, help='Kernel size for dilate')
    parser.add_argument('-i', '--iter', type=int, default=1, help='Iterations for dilate')
    args = parser.parse_args()

    if not os.path.exists(args.source_dir):
        print('Input dir does not exist: {}'.format(args.source_dir))
        exit()

    IN_SUBDIR = 'orig-input-depth'
    OUT_SUBDIR = 'output-depth'
    INT_SUBDIR = 'intermediate-results'
    EXT_IN_DEPTH = '-input-depth.png'
    EXT_OUT_DEPTH = '-output-depth.png'
    EXT_MASK = '-mask.png'

    MASK_INVALID_SUBDIR = 'masks_invalid_live_demo'
    EXT_MASK_INVALID = '-rgb.png'

    CLEANED_OUT_SUBDIR = 'output-depth-cleaned'
    EXT_CLEANED_OUT_DEPTH = '-output-depth.png'
    EXT_CLEANED_IN_DEPTH = '-input-depth.png'

    in_depth_files_list = sorted(glob.glob(os.path.join(args.source_dir, IN_SUBDIR, '*' + EXT_IN_DEPTH)))
    out_depth_files_list = sorted(glob.glob(os.path.join(args.source_dir, OUT_SUBDIR, '*' + EXT_OUT_DEPTH)))
    mask_files_list = sorted(glob.glob(os.path.join(args.source_dir, INT_SUBDIR, '*' + EXT_MASK)))
    mask_invalid_files_list = sorted(glob.glob(os.path.join(args.source_dir, MASK_INVALID_SUBDIR, '*' + EXT_MASK_INVALID)))
    if len(in_depth_files_list) == 0:
        raise ValueError('No files in source dir {}'.format(in_depth_files_list))
    if len(out_depth_files_list) == 0:
        raise ValueError('No files in source dir {}'.format(out_depth_files_list))

    output_dir = os.path.join(args.source_dir, CLEANED_OUT_SUBDIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for iii, sample in enumerate(tqdm.tqdm(zip(in_depth_files_list, out_depth_files_list, mask_files_list))):
        in_d_file, out_d_file, mask_t_file = sample

        in_depth_image = imageio.imread(in_d_file)
        out_depth_image = imageio.imread(out_d_file)
        mask_trans = imageio.imread(mask_t_file)

        # if len(mask_trans.shape) > 2:
        #     print('mask_trans:', mask_trans.shape)
        #     mask_trans2 = np.array(mask_trans[:, :, 0])
        #     print('mask_trans:', mask_trans.shape, mask_trans.dtype, mask_trans.min(), mask_trans.max())

        #     clean_mask_file = os.path.join(args.source_dir, INT_SUBDIR, '{:09d}'.format(iii) + '-mask-cleaned.png')
        #     print('clean_mask_file:', clean_mask_file)
        #     imageio.imwrite(mask_t_file, mask_trans2)

        mask = ((in_depth_image > 0) * 255).astype(np.uint8)
        kernel = np.ones((args.kernel, args.kernel), np.uint8)
        mask_d = cv2.dilate(mask, kernel, iterations=args.iter)

        mask_d[mask_trans > 0] = 255

        out_depth_image[mask_d == 0] = 0
        in_depth_image[mask_d == 0] = 0

        if iii > 53:
            mask_invalid = imageio.imread(os.path.join(args.source_dir, MASK_INVALID_SUBDIR, '{:09d}'.format(iii)+EXT_MASK_INVALID))
            kernel = np.ones((args.kernel, args.kernel), np.uint8)
            mask_invalid = cv2.dilate(mask_invalid, kernel, iterations=args.iter)
            out_depth_image[mask_invalid > 100] = 0
            in_depth_image[mask_invalid > 100] = 0


        # cv2.namedWindow('Orig Input Depth')
        # cv2.imshow('Orig Input Depth', mask)
        # cv2.namedWindow('Dilated Input Depth')
        # cv2.imshow('Dilated Input Depth', mask_d)
        # cv2.waitKey()

        imageio.imwrite(os.path.join(output_dir, '{:09d}'.format(iii) + EXT_CLEANED_OUT_DEPTH), out_depth_image)
        imageio.imwrite(os.path.join(output_dir, '{:09d}'.format(iii) + EXT_CLEANED_IN_DEPTH), in_depth_image)
        # print('Cleaned image {}'.format(iii))
