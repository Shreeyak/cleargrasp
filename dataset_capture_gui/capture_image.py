"""Script to capture images for the dataset of real objects from a realsense D400 series camera
First, we place transparent objects, capture it's RGB image, then replace with opaque objects,
capture it's depth image."""
#!/usr/bin/env python

import argparse
import glob
import io
import os
import time
import subprocess
import sys

import imageio
import numpy as np
import torch
import torch.nn as nn
import yaml
import cv2
from PIL import Image
from attrdict import AttrDict
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import utils as api_utils
from live_demo.realsense.camera import Camera


def blend_2_images(img1, img2, p1, p2):
    """Blends together 2 images with 50% transparency each

    Args:
        img1 (numpy.ndarray): RGB image, shape H,W,3
        img2 (numpy.ndarray): RGB image, shape H,W,3
    """
    dst = cv2.addWeighted(img1, p1, img2, p2, 0)

    return dst

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run GUI app for capturing real dataset using realsense camera')
    parser.add_argument('-c', '--continue_exp', action='store_true', help='Should continue with prev save folder?')
    parser.add_argument('-t', '--height', type=int, default=1080, help='Should continue with prev save folder?')
    parser.add_argument('-w', '--width', type=int, default=1920, help='Should continue with prev save folder?')
    args = parser.parse_args()

    ###################### GET IMAGES FROM CAMERA #############################
    print('Displaying output images (left-to-right, top-to-bottom): \nInput image, Predicted Surface Normals,',
          'Predicted Outlines, Input Depth, Output Depth, Blank Image\n')
    rcamera = Camera()
    time.sleep(1)  # Give camera some time to load data

    # Create directory to save captures
    captures_dir_root = 'data/captures'
    runs = sorted(glob.glob(os.path.join(captures_dir_root, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    captures_dir = os.path.join(captures_dir_root, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(captures_dir):
        if len(os.listdir(captures_dir)) > 1 and args.continue_exp is False:
            captures_dir = os.path.join(captures_dir_root, 'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(captures_dir)
    else:
        os.makedirs(captures_dir)

    print('Saving captured images to folder: ' + colored('"{}"'.format(captures_dir), 'blue'))
    print('\n Press "c" to capture and save image, press "q" to quit\n')

    # Get params from camera
    camera_intrinsics = rcamera.color_intr
    realsense_fx = camera_intrinsics[0, 0]
    realsense_fy = camera_intrinsics[1, 1]
    realsense_cx = camera_intrinsics[0, 2]
    realsense_cy = camera_intrinsics[1, 2]
    with open(os.path.join(captures_dir, "camera_parameters.txt"), "w") as file1:
        L = ["Image Height: {}\n".format(rcamera.im_height),
             "Image Width: {}\n".format(rcamera.im_width),
             "fx: {}\n".format(realsense_fx),
             "fy: {}\n".format(realsense_fy),
             "cx: {}\n".format(realsense_cx),
             "cy: {}\n".format(realsense_cy)]
        file1.writelines(L)

    display_height = args.height
    display_width = args.width
    color_img_transparent = None
    depth_img_transparent = None
    color_img_opaque = None
    depth_img_opaque = None

    transparent_captured = False
    opaque_captured = False
    state = 'normal'
    transparent_blend = 0.5
    opaque_blend = 0.5

    capture_num = 0  # Used to name images captured
    if args.continue_exp is True:
        filenames = sorted(os.listdir(captures_dir))
        if len(filenames) > 0:
            last_index = filenames.pop()[:9]
            while not last_index.isnumeric():
                last_index = filenames.pop()[:9]
            capture_num = int(last_index) + 1

    counter = 0
    while True:
        color_img, depth_img = rcamera.get_data()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        depth_img = depth_img.astype(np.float32)

        if transparent_captured is False and opaque_captured is False:
            grid_image = color_img.copy()
            cv2.putText(grid_image, "Waiting", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif transparent_captured is True and opaque_captured is False:
            grid_image = blend_2_images(color_img_transparent, color_img, transparent_blend, opaque_blend)
            cv2.putText(grid_image, "Transparent Saved", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif transparent_captured is False and opaque_captured is True:
            grid_image = blend_2_images(color_img, color_img_opaque, transparent_blend, opaque_blend)
            cv2.putText(grid_image, "Opaque Saved", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif transparent_captured is True and opaque_captured is True:
            grid_image = blend_2_images(color_img_transparent, color_img_opaque, transparent_blend, opaque_blend)
            cv2.putText(grid_image, "Both Saved. Press spacebar to Confirm", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        grid_image = cv2.resize(grid_image, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Frame_capture_dataset', grid_image)

        keypress = cv2.waitKey(10) & 0xFF
        if keypress == ord('q'):
            # Quit, if 'q' key pressed
            break

        elif keypress == ord('r'):
            # Reset state
            transparent_captured = False
            opaque_captured = False

        elif keypress == ord('t'):
            # Reset transparent state
            transparent_captured = not transparent_captured

        elif keypress == ord('y'):
            # Reset opaque state
            opaque_captured = not opaque_captured

        elif keypress == ord('n'):
            # Show only transparent object
            transparent_blend = 1.0
            opaque_blend = 0
        elif keypress == ord('m'):
            # Show only opaque object
            transparent_blend = 0.0
            opaque_blend = 1.0
        elif keypress == ord(','):
            # Show 50% blend of both
            transparent_blend = 0.5
            opaque_blend = 0.5

        elif keypress == ord('c'):
            # Get RGB Image
            transparent_captured = True

            color_img_transparent = color_img
            depth_img_transparent = depth_img
            print('captured rgb image')

        elif keypress == ord('v'):
            # Get Depth Image
            opaque_captured = True

            color_img_opaque = color_img
            depth_img_opaque = depth_img

            # Temporal Filtering - 8 images deep
            depth_img_list = []
            for i in range(30):
                color_img_tmp, depth_img_tmp = rcamera.get_data()
                depth_img_tmp = depth_img_tmp.astype(np.float32)
                depth_img_list.append(depth_img_tmp)
            depth_img_stack = np.stack(depth_img_list, axis=0)  # Get (N, H, W) array
            depth_nonzero = np.count_nonzero(depth_img_stack, axis=0)
            depth_img_sum = np.sum(depth_img_stack, axis=0)
            depth_img_opaque_filtered = depth_img_sum / depth_nonzero
            print('captured depth image')

        elif keypress == ord(' '):
            # Save Image Pair
            if transparent_captured is True and opaque_captured is True:
                transparent_captured = False
                opaque_captured = False

                imageio.imwrite(os.path.join(captures_dir, '{0:09d}-transparent-rgb-img.jpg'.format(capture_num)), cv2.cvtColor(color_img_transparent, cv2.COLOR_BGR2RGB), quality=100)
                api_utils.exr_saver(os.path.join(captures_dir, '{0:09d}-opaque-depth-img.exr'.format(capture_num)), depth_img_opaque_filtered, ndim=3)
                imageio.imwrite(os.path.join(captures_dir, '{0:09d}-opaque-rgb-img.jpg'.format(capture_num)), cv2.cvtColor(color_img_opaque, cv2.COLOR_BGR2RGB), quality=100)
                api_utils.exr_saver(os.path.join(captures_dir, '{0:09d}-transparent-depth-img.exr'.format(capture_num)), depth_img_transparent, ndim=3)

                print('saved image pair {0:06d}'.format(capture_num))
                capture_num += 1
            else:
                print(colored('Make sure both RGB and Depth image are captured.', 'red'))

    # Closes all the frames
    cv2.destroyAllWindows()
