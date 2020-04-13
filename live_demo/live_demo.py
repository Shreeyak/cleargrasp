#!/usr/bin/env python3
'''Live demo of ClearGrasp
Will predict depth for all transparent objects on images streaming from a realsense camera using our API.
'''

import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import h5py
import numpy as np
import numpy.ma as ma
import termcolor
import yaml
from attrdict import AttrDict
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import utils, depth_completion_api
from realsense import camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run live demo of depth completion on realsense camera')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config.yaml')
    args = parser.parse_args()

    # Initialize Camera
    print('Running live demo of depth completion. Make sure realsense camera is streaming.\n')
    rcamera = camera.Camera()
    camera_intrinsics = rcamera.color_intr
    realsense_fx = camera_intrinsics[0, 0]
    realsense_fy = camera_intrinsics[1, 1]
    realsense_cx = camera_intrinsics[0, 2]
    realsense_cy = camera_intrinsics[1, 2]
    time.sleep(1)  # Give camera some time to load data

    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)

    # Create directory to save captures
    runs = sorted(glob.glob(os.path.join(config.captures_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    captures_dir = os.path.join(config.captures_dir, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(captures_dir):
        if len(os.listdir(captures_dir)) > 5:
            # Min 1 file always in folder: copy of config file
            captures_dir = os.path.join(config.captures_dir, 'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(captures_dir)
    else:
        os.makedirs(captures_dir)

    # Save a copy of config file in the logs
    shutil.copy(CONFIG_FILE_PATH, os.path.join(captures_dir, 'config.yaml'))

    print('Saving captured images to folder: ' + termcolor.colored('"{}"'.format(captures_dir), 'blue'))
    print('\n Press "c" to capture and save image, press "q" to quit\n')

    # Initialize Depth Completion API
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)
    depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=config.normals.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                normalsModel=config.normals.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.normals.inferenceHeight,
                                                                normalsInferenceWidth=config.normals.inferenceWidth,
                                                                outlinesInferenceHeight=config.normals.inferenceHeight,
                                                                outlinesInferenceWidth=config.normals.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=captures_dir)
    capture_num = 0
    while True:
        # Get Frame. Expected format: ColorImg -> (H, W, 3) uint8, DepthImg -> (H, W) float64
        color_img, input_depth = rcamera.get_data()
        input_depth = input_depth.astype(np.float32)

        try:
            output_depth, filtered_output_depth = depthcomplete.depth_completion(
                color_img,
                input_depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode)
        except depth_completion_api.DepthCompletionError as e:
            print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
            continue

        color_img = depthcomplete.input_image
        input_depth = depthcomplete.input_depth
        surface_normals = depthcomplete.surface_normals
        surface_normals_rgb = depthcomplete.surface_normals_rgb
        occlusion_weight = depthcomplete.occlusion_weight
        occlusion_weight_rgb = depthcomplete.occlusion_weight_rgb
        outlines_rgb = depthcomplete.outlines_rgb

        # Display Results in Window
        input_depth_mapped = utils.depth2rgb(input_depth, min_depth=config.depthVisualization.minDepth,
                                             max_depth=config.depthVisualization.maxDepth,
                                             color_mode=cv2.COLORMAP_JET, reverse_scale=True)
        output_depth_mapped = utils.depth2rgb(output_depth, min_depth=config.depthVisualization.minDepth,
                                              max_depth=config.depthVisualization.maxDepth,
                                              color_mode=cv2.COLORMAP_JET, reverse_scale=True)
        filtered_output_depth_mapped = utils.depth2rgb(filtered_output_depth,
                                                       min_depth=config.depthVisualization.minDepth,
                                                       max_depth=config.depthVisualization.maxDepth,
                                                       color_mode=cv2.COLORMAP_JET, reverse_scale=True)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        surface_normals_rgb = cv2.cvtColor(surface_normals_rgb, cv2.COLOR_RGB2BGR)
        outlines_rgb = cv2.cvtColor(outlines_rgb, cv2.COLOR_RGB2BGR)
        occlusion_weight_rgb = cv2.cvtColor(occlusion_weight_rgb, cv2.COLOR_RGB2BGR)

        grid_image1 = np.concatenate((color_img, surface_normals_rgb, outlines_rgb, occlusion_weight_rgb), 1)
        grid_image2 = np.concatenate((input_depth_mapped, output_depth_mapped, filtered_output_depth_mapped,
                                      np.zeros(color_img.shape, dtype=color_img.dtype)), 1)
        grid_image = np.concatenate((grid_image1, grid_image2), 0)

        cv2.imshow('Live Demo', grid_image)
        keypress = cv2.waitKey(10) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            # Save captured data to config.captures_dir
            depthcomplete.store_depth_completion_outputs(captures_dir,
                                                         capture_num,
                                                         min_depth=config.depthVisualization.minDepth,
                                                         max_depth=config.depthVisualization.maxDepth)
            print('captured image {0:06d}'.format(capture_num))
            capture_num += 1

    cv2.destroyAllWindows()
