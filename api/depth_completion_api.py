"""Depth Completion API
Exposes class containing API for running depth completion using depth2depth module (external C++ executable).
depth2depth is taken from the project DeepCompletion. See: http://deepcompletion.cs.princeton.edu/
"""
import errno
import json
import os
import struct
import subprocess
import sys
import time
import warnings

# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d

import cv2
import imageio
import h5py
import numpy as np
import scipy
import torch
import torch.nn as nn
from PIL import Image
from . import inference_models, utils


class DepthCompletionError(Exception):
    """Exception raised when the depth2depth module fails to complete depth.

    Args:
        msg(str): Explanation of the error
    """

    def __init__(self, msg='Depth Completion Failed!'):
        self.msg = msg

    def __str__(self):
        return (repr(self.msg))


class DepthToDepthCompletion(object):
    """This is an API for running depth completion using depth2depth module (external C++ executable).

    The depth2depth module is an external C++ executable that accepts an input depth with holes and fills in the
    holes using a global optimization algorithm. It requires 3 files as inputs:
        - input depth
        - surface normals
        - occlusion weights.
    The depth2depth module returns an output depth which is saved to disk as well.

    Surface normals and occlusion weights are estimated using 2 deep learning models.

    Attributes:
        normalsWeightsFile (str): Path to the weights file for surface normals prediction model.
        outlinesWeightsFile (str): Path to the weights file for outlines prediction model.
        masksWeightsFile (str): Path to the weights file for masks prediction model
                               (semantic segmentation of transparent objects).
                               NOTE: If not using any segmentation model, pass empty string
        normalsModel (str): Default: 'deeplab_resnet'. Which model the suface normals model
                            uses: ['unet', 'deeplab_resnet']
        outlinesModel (str): Default: 'deeplab_resnet'. Which model the outlines model uses: ['unet', 'deeplab_resnet']
        masksModel (str): Default: 'deeplab_resnet'. Which model the outlines model uses: ['', deeplab_resnet']
                          NOTE: If not using any segmentation model (masksWeightsFile is empty string),
                                this attribute has no effect.

        depth2depthExecutable (str): Default: './depth2depth'. Path to the depth2depth executable.
        outputImgHeight (int): Default: 288. Pipeline will output images with this height.
        outputImgWidth (int): Default: 512. Pipeline will output images with this width.
        fx (int): Default: 370. The focal len along x-axis in pixels of camera used to capture image.
        fy (int): Default: 370. The focal len along y-axis in pixels of camera used to capture image.
        cx (int): Default: 256. The center of the image (along x-axis) as per camera used to capture image.
        cy (int): Default: 144. The center of the image (along y-axis) as per camera used to capture image.

        filter_d (int): Default: 0. Param of bilateral filer for output depth.
                        Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is
                        computed from sigmaSpace .
        filter_sigmaColor (int): Default: 100. Param of bilateral filer for output depth.
                           Filter sigma in the color space. A larger value of the parameter means that farther colors
                           within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger
                           areas of semi-equal color.
        filter_sigmaSpace (int): Default: 100. Param of bilateral filer for output depth.
                           Filter sigma in the coordinate space. A larger value of the parameter means that farther
                           pixels will influence each other as long as their colors are close enough (see sigmaColor ).
                           When d>0 , it specifies the neighborhood size regardless of sigmaSpace .
                           Otherwise, d is proportional to sigmaSpace.
        maskinferenceHeight (int): Height of image during inference for model of masks.
        maskinferenceWidth (int): Width of image during inference for model of masks.
        normalsInferenceHeight (int): Height of image during inference for model of surface normals.
        normalsInferenceWidth (int): Width of image during inference for model of surface normals.
        outlinesInferenceHeight (int): Height of image during inference for model of outlines.
        outlinesInferenceWidth (int): Width of image during inference for model of outlines.
        min_depth (float): The min depth to be considered while visualizing depth and constructing point clouds.
        max_depth (float): The max depth to be considered while visualizing depth and constructing point clouds.
        tmp_dir (str): The dir where tmp intermediate outputs like surface normals and outlines are stored.

    """

    # Folders for storing results of depth completion
    FOLDER_MAP = {
        'orig-input-depth': {
            'postfix': '-input-depth.exr',
            'folder-name': 'orig-input-depth'
        },
        'orig-input-point-cloud': {
            'postfix': '-input-pointcloud.ply',
            'folder-name': 'orig-input-point-cloud'
        },
        'input-depth': {
            'postfix': '-input-depth.exr',
            'folder-name': 'input-depth'
        },
        'input-image': {
            'postfix': '-rgb.png',
            'folder-name': 'input-image'
        },
        'input-point-cloud': {
            'postfix': '-input-pointcloud.ply',
            'folder-name': 'input-point-cloud'
        },
        'output-depth': {
            'postfix': '-output-depth.exr',
            'folder-name': 'output-depth'
        },
        'output-point-cloud': {
            'postfix': '-output-pointcloud.ply',
            'folder-name': 'output-point-cloud'
        },
        'gt-depth': {
            'postfix': '-gt-depth.exr',
            'folder-name': 'gt-depth'
        },
        'gt-point-cloud': {
            'postfix': '-gt-pointcloud.ply',
            'folder-name': 'gt-point-cloud'
        },
        'masks': {
            'postfix': '-mask.png',
            'folder-name': 'intermediate-results'
        },
        'normals': {
            'postfix': '-normal-rgb.png',
            'folder-name': 'intermediate-results'
        },
        'outlines': {
            'postfix': '-outline-weight.png',
            'folder-name': 'intermediate-results'
        },
        'result-viz': {
            'postfix': '-result-viz.png',
            'folder-name': 'result-viz'
        },
    }

    _TMP_DIR = 'data/tmp'

    def __init__(
            self,
            normalsWeightsFile,
            outlinesWeightsFile,
            masksWeightsFile='',
            normalsModel='deeplab_resnet',
            outlinesModel='deeplab_resnet',
            masksModel='deeplab_resnet',
            depth2depthExecutable='./depth2depth',
            outputImgHeight=288,
            outputImgWidth=512,
            fx=370,
            fy=370,
            cx=256,
            cy=144,
            filter_d=0,
            filter_sigmaColor=100,
            filter_sigmaSpace=100,
            maskinferenceHeight=288,
            maskinferenceWidth=512,
            normalsInferenceHeight=288,
            normalsInferenceWidth=512,
            outlinesInferenceHeight=288,
            outlinesInferenceWidth=512,
            min_depth=0.0,
            max_depth=3.0,
            tmp_dir=None,
    ):

        self.depth2depthExecutable = depth2depthExecutable
        self.outputImgHeight = outputImgHeight
        self.outputImgWidth = outputImgWidth
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = filter_d
        self.sigmaColor = filter_sigmaColor
        self.sigmaSpace = filter_sigmaSpace
        self.masksWeightsFile = masksWeightsFile
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Paths to the intermediate files generated for depth2depth
        if tmp_dir is not None:
            self._TMP_DIR = tmp_dir
        self._PATH_TMP_INPUT_DEPTH = os.path.join(self._TMP_DIR, 'input-depth.png')
        self._PATH_TMP_SURFACE_NORMALS = os.path.join(self._TMP_DIR, 'predicted-surface-normals.h5')
        self._PATH_TMP_OCCLUSION_WEIGHTS = os.path.join(self._TMP_DIR, 'predicted-occlusion-weight.png')
        self._PATH_TMP_OUTPUT_DEPTH = os.path.join(self._TMP_DIR, 'output-depth.png')
        self.input_image = None
        self.orig_input_depth = None
        self.input_depth = None
        self.output_depth = None
        self.filtered_output_depth = None
        self.surface_normals = None
        self.surface_normals_rgb = None
        self.occlusion_weight = None
        self.occlusion_weight_rgb = None
        self.outlines_rgb = None
        self.mask_predicted = None
        self.mask_valid_region = None

        if not os.path.isdir(self._TMP_DIR):
            os.makedirs(self._TMP_DIR)

        self.inferenceNormals = inference_models.InferenceNormals(normalsWeightsFile=normalsWeightsFile,
                                                                  normalsModel=normalsModel,
                                                                  imgHeight=normalsInferenceHeight,
                                                                  imgWidth=normalsInferenceWidth)
        self.inferenceOutlines = inference_models.InferenceOutlines(outlinesWeightsFile=outlinesWeightsFile,
                                                                    outlinesModel=outlinesModel,
                                                                    imgHeight=outlinesInferenceHeight,
                                                                    imgWidth=outlinesInferenceWidth)
        # Model for masks is only used when predicted masks are used for depth completion
        if masksWeightsFile:
            self.inferenceMasks = inference_models.InferenceMasks(masksWeightsFile=masksWeightsFile,
                                                                  masksModel=masksModel,
                                                                  imgHeight=maskinferenceHeight,
                                                                  imgWidth=maskinferenceWidth)

    def estimate_normals_write_ptcloud(self, filename, xyz_points, height, width):
        """Uses Open3D to estimate normals
        Due to problems between Open3D and Pytorch, the import statement is inserted here locally.

        Args:
            filename (str): Path of filename where point cloud will be saved
            xyz_points (numpy.ndarray): The XYZ location of each pixel. Shape: (num of pixels, 3)
            height (int): Height of original image
            width (int): Width of original image
        Returns:
            numpy.ndarray: Surface Normal
            numpy.ndarray: Surface Normal RGB

        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10000))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        pcd.normalize_normals()
        pcd_normals = np.asarray(pcd.normals)
        pcd_normals[:, 1] = pcd_normals[:, 1] * -1.0  # Orient normals according to (Y-Up, X-Right)
        pcd_normals[:, 2] = pcd_normals[:, 2] * -1.0  # Orient normals according to (Y-Up, X-Right)

        rgb_points = (((pcd_normals + 1) / 2) * 255).astype(np.uint8)

        # Write header of .ply file
        with open(filename, 'wb') as fid:
            fid.write(bytes('ply\n', 'utf-8'))
            fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
            fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
            fid.write(bytes('property float x\n', 'utf-8'))
            fid.write(bytes('property float y\n', 'utf-8'))
            fid.write(bytes('property float z\n', 'utf-8'))
            fid.write(bytes('property uchar red\n', 'utf-8'))
            fid.write(bytes('property uchar green\n', 'utf-8'))
            fid.write(bytes('property uchar blue\n', 'utf-8'))
            fid.write(bytes('end_header\n', 'utf-8'))

            # Write 3D points to .ply file
            for i in range(xyz_points.shape[0]):
                fid.write(
                    bytearray(
                        struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                    rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(),
                                    rgb_points[i, 2].tostring())))

        return

    def store_depth_completion_outputs(self, root_dir, files_prefix, min_depth=0.0, max_depth=1.5):
        """Writes the outputs of depth completion to files
        It stores:
            - Resized Orig Input Depth Image
            - Point cloud of orig input depth
            - Input Depth Image - The orig input depth modified by chosen method
            - Point cloud of input depth image
            - Output Depth Image
            - Point cloud of output depth
            - Point cloud of ground truth data, colored in surface normals estimation
            # - Filtered Depth Image
            # - Point cloud of filtered output depth
            - Results Viz: A collage of the results of depth completion, including visualization of predictions of
                           surface normals and outlines.

        Args:
            root_dir (str): Path to dir to store the data
            files_prefix (int): Prefix to add to the file names
            min_depth (float): Min depth for visualization of depth images in results (in meters)
            max_depth (float): Max depth for visualization of depth images in results (in meters)
            mask_valid_region (numpy.ndarray): A mask of the valid regions in depth image, that has been used
                to calc the metric.

        Returns:
            float32: Mean Error on reconstructed output depth (only in case of Synthetic data, where input depth is ground truth)
            float32: Mean Error on reconstructed filtered output depth (only in case of Synthetic data, where input depth is ground truth)
        """

        # Create dirs
        orig_input_depth_dir = os.path.join(root_dir, self.FOLDER_MAP['orig-input-depth']['folder-name'])
        orig_input_ptcloud_dir = os.path.join(root_dir, self.FOLDER_MAP['orig-input-point-cloud']['folder-name'])
        input_depth_dir = os.path.join(root_dir, self.FOLDER_MAP['input-depth']['folder-name'])
        input_ptcloud_dir = os.path.join(root_dir, self.FOLDER_MAP['input-point-cloud']['folder-name'])
        output_depth_dir = os.path.join(root_dir, self.FOLDER_MAP['output-depth']['folder-name'])
        output_ptcloud_dir = os.path.join(root_dir, self.FOLDER_MAP['output-point-cloud']['folder-name'])
        results_viz_dir = os.path.join(root_dir, self.FOLDER_MAP['result-viz']['folder-name'])
        masks_dir = os.path.join(root_dir, self.FOLDER_MAP['masks']['folder-name'])
        normals_dir = os.path.join(root_dir, self.FOLDER_MAP['normals']['folder-name'])
        outlines_dir = os.path.join(root_dir, self.FOLDER_MAP['outlines']['folder-name'])
        input_image_dir = os.path.join(root_dir, self.FOLDER_MAP['input-image']['folder-name'])
        gt_depth_dir = os.path.join(root_dir, self.FOLDER_MAP['gt-depth']['folder-name'])
        gt_ptcloud_dir = os.path.join(root_dir, self.FOLDER_MAP['gt-point-cloud']['folder-name'])

        dir_list = [os.path.join(root_dir, self.FOLDER_MAP[key]['folder-name']) for key in self.FOLDER_MAP]
        for dirname in dir_list:
            try:
                os.mkdir(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        # Save Input Image
        input_image_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['input-image']['postfix']
        input_image_filename = os.path.join(input_image_dir, input_image_filename)
        imageio.imwrite(input_image_filename, self.input_image)

        # Save Ground Truth depth and point cloud.
        gt_depth_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['gt-depth']['postfix']
        gt_depth_filename = os.path.join(gt_depth_dir, gt_depth_filename)
        utils.exr_saver(gt_depth_filename, self.depth_gt, ndim=3)

        gt_ptcloud_filename = ('{:09d}'.format(files_prefix) + self.FOLDER_MAP['gt-point-cloud']['postfix'])
        gt_ptcloud_filename = os.path.join(gt_ptcloud_dir, gt_ptcloud_filename)
        xyz_points, rgb_points = utils._get_point_cloud(self.input_image, self.depth_gt, self.fx, self.fy, self.cx,
                                                        self.cy)
        self.estimate_normals_write_ptcloud(gt_ptcloud_filename, xyz_points, self.outputImgHeight, self.outputImgWidth)

        # Save Orig input depth and point cloud
        input_depth_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['orig-input-depth']['postfix']
        input_depth_filename = os.path.join(orig_input_depth_dir, input_depth_filename)
        utils.exr_saver(input_depth_filename, self.orig_input_depth, ndim=3)

        input_ptcloud_filename = ('{:09d}'.format(files_prefix) + self.FOLDER_MAP['orig-input-point-cloud']['postfix'])
        input_ptcloud_filename = os.path.join(orig_input_ptcloud_dir, input_ptcloud_filename)
        # utils.write_point_cloud(input_ptcloud_filename, self.input_image, self.orig_input_depth, self.fx, self.fy,
        #                         self.cx, self.cy)
        xyz_points, rgb_points = utils._get_point_cloud(self.input_image, self.orig_input_depth, self.fx, self.fy,
                                                        self.cx, self.cy)
        self.estimate_normals_write_ptcloud(input_ptcloud_filename, xyz_points, self.outputImgHeight,
                                            self.outputImgWidth)

        # Save input depth and point cloud
        input_depth_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['input-depth']['postfix']
        input_depth_filename = os.path.join(input_depth_dir, input_depth_filename)
        utils.exr_saver(input_depth_filename, self.input_depth, ndim=3)

        input_ptcloud_filename = ('{:09d}'.format(files_prefix) + self.FOLDER_MAP['input-point-cloud']['postfix'])
        input_ptcloud_filename = os.path.join(input_ptcloud_dir, input_ptcloud_filename)
        utils.write_point_cloud(input_ptcloud_filename, self.input_image, self.input_depth, self.fx, self.fy, self.cx,
                                self.cy)

        # Save output depth and point cloud
        output_depth_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['output-depth']['postfix']
        output_depth_filename = os.path.join(output_depth_dir, output_depth_filename)
        utils.exr_saver(output_depth_filename, self.output_depth, ndim=3)

        output_ptcloud_filename = ('{:09d}'.format(files_prefix) + self.FOLDER_MAP['output-point-cloud']['postfix'])
        output_ptcloud_filename = os.path.join(output_ptcloud_dir, output_ptcloud_filename)
        utils.write_point_cloud(output_ptcloud_filename, self.surface_normals_rgb, self.output_depth, self.fx, self.fy,
                                self.cx, self.cy)

        # Store Masks
        mask_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['masks']['postfix']
        mask_filename = os.path.join(masks_dir, mask_filename)
        imageio.imwrite(mask_filename, self.mask_predicted)

        # Store Normals
        normal_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['normals']['postfix']
        normal_filename = os.path.join(normals_dir, normal_filename)
        imageio.imwrite(normal_filename, self.surface_normals_rgb)

        # Store Outlines
        outline_filename = '{:09d}'.format(files_prefix) + self.FOLDER_MAP['outlines']['postfix']
        outline_filename = os.path.join(outlines_dir, outline_filename)
        imageio.imwrite(outline_filename, self.occlusion_weight_rgb)

        # STORE COLLAGE OF RESULTS - RGB IMAGE, SURFACE NORMALS, OCCLUSION WEIGHTS, DEPTH IMAGES
        # create RGB visualization  of depth images
        COLOR_MAP = cv2.COLORMAP_JET
        input_depth_rgb = utils.depth2rgb(self.input_depth,
                                          min_depth=min_depth,
                                          max_depth=max_depth,
                                          color_mode=COLOR_MAP)
        orig_input_depth_rgb = utils.depth2rgb(self.orig_input_depth,
                                               min_depth=min_depth,
                                               max_depth=max_depth,
                                               color_mode=COLOR_MAP)
        output_depth_rgb = utils.depth2rgb(self.output_depth,
                                           min_depth=min_depth,
                                           max_depth=max_depth,
                                           color_mode=COLOR_MAP)
        gt_depth_rgb = utils.depth2rgb(self.depth_gt, min_depth=min_depth, max_depth=max_depth, color_mode=COLOR_MAP)

        # Store input-output depth RGB
        gt_depth_filename = '{:09d}'.format(files_prefix) + '-gt-depth-rgb.png'
        gt_depth_filename = os.path.join(gt_depth_dir, gt_depth_filename)
        imageio.imwrite(gt_depth_filename, gt_depth_rgb)
        input_depth_filename = '{:09d}'.format(files_prefix) + '-input-depth-rgb.png'
        input_depth_filename = os.path.join(orig_input_depth_dir, input_depth_filename)
        imageio.imwrite(input_depth_filename, orig_input_depth_rgb)
        input_depth_filename = '{:09d}'.format(files_prefix) + '-modified-input-depth-rgb.png'
        input_depth_filename = os.path.join(input_depth_dir, input_depth_filename)
        imageio.imwrite(input_depth_filename, input_depth_rgb)
        output_depth_filename = '{:09d}'.format(files_prefix) + '-output-depth-rgb.png'
        output_depth_filename = os.path.join(output_depth_dir, output_depth_filename)
        imageio.imwrite(output_depth_filename, output_depth_rgb)

        # Calculate error in output depth in meters (only on pixels that were 0.0 in input depth)
        DEPTH_SCALING_M_TO_CM = 100
        error_output_depth = np.abs(self.output_depth - self.orig_input_depth) * DEPTH_SCALING_M_TO_CM
        mask = (self.input_depth != 0.0)
        error_output_depth = np.ma.masked_array(error_output_depth, mask=mask)
        error_output_depth = round(error_output_depth.mean(), 4)
        error_filtered_output_depth = np.abs(self.filtered_output_depth - self.orig_input_depth) * DEPTH_SCALING_M_TO_CM
        error_filtered_output_depth = np.ma.masked_array(error_filtered_output_depth, mask=mask)
        error_filtered_output_depth = round(error_filtered_output_depth.mean(), 4)

        # Write error onto output depth rgb image
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(output_depth_rgb, 'Error: {:.2f}cm'.format(error_output_depth), (120, 270), font, 1,
        #             (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(filtered_output_depth_rgb, 'Error: {:.2f}cm'.format(error_filtered_output_depth), (120, 270), font,
        #             1, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay Mask on Img
        mask_rgb = self.input_image.copy()
        mask_rgb[self.mask_predicted > 0, 0] = 255
        mask_rgb[self.outlines_rgb[:, :, 1] > 0, 1] = 255
        masked_img = cv2.addWeighted(mask_rgb, 0.6, self.input_image, 0.4, 0)

        mask_valid_region_3d = np.stack([self.mask_valid_region] * 3, axis=2)  # Ground truth mask - invalid depth pixels

        # Create Vizualization of all the results
        grid_image1 = np.concatenate(
            (self.input_image, self.surface_normals_rgb, self.outlines_rgb, self.occlusion_weight_rgb, masked_img), 1)
        grid_image2 = np.concatenate(
            (orig_input_depth_rgb, input_depth_rgb, output_depth_rgb, gt_depth_rgb, mask_valid_region_3d), 1)
        grid_image = np.concatenate((grid_image1, grid_image2), 0)

        path_result_viz = os.path.join(results_viz_dir,
                                       '{:09d}'.format(files_prefix) + self.FOLDER_MAP['result-viz']['postfix'])
        imageio.imwrite(path_result_viz, grid_image)

        return error_output_depth, error_filtered_output_depth

    def _modify_synthetic_depth_delete_objects(self, input_depth, json_file_path, variant_mask_path, centroid_width=5):
        """Synthetic data - Create mask of transparent objects, except for a small point at the centroid of each object

        This method is used to see if depth completion works in principle if a single point of depth is given
        per tranparent object. We create a mask of each object and use it to mask out the depth of all objects in a
        depth image, except for a small point at the centroid of each object.

        Args:
            input_depth (numpy.ndarray): Depth Image. dtype=float32, shape=(H, W)
            json_file_path (str): Path to the json file (contains info about image and masks)
            variant_mask_path (str): Path to the variant mask of the image (contains masks of each object variant)
                                     Shape=(H, W)
            centroid_width (int): Width of hole in mask at centroid of objects

        Returns:
            numpy.ndarray: Depth Image with each instance of an object masked out, except some pixels at their
                           centroids. dtype=np.uint8

        Raises:
            ZeroDivisionError: Error in calculating centroid due to division by zero
        """
        # Load variant mask
        variant_mask = utils.exr_loader(variant_mask_path, ndim=1)
        maskWidth = input_depth.shape[1]
        maskHeight = input_depth.shape[0]
        variant_mask = cv2.resize(variant_mask, (maskWidth, maskHeight), interpolation=cv2.INTER_NEAREST)

        # Read json file
        json_file = open(json_file_path)
        data = json.load(json_file)
        object_id = []
        for key, values in data['variants']['masks_and_poses_by_pixel_value'].items():
            object_id.append(key)

        # Create and combine masks for each transparent object in image
        final_mask = np.zeros(variant_mask.shape, dtype=np.uint8)

        for i in range(len(object_id)):
            mask = np.zeros(variant_mask.shape, dtype=np.uint8)
            mask[variant_mask == int(object_id[i])] = 255

            # calculate Centroid of mask
            M = cv2.moments(mask)

            try:
                cY = int(M["m10"] / M["m00"])
                cX = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                warnings.warn('Failed to calculate centroid of an object when creating mask')
                continue

            mask[(cX - centroid_width):(cX + centroid_width), (cY - centroid_width):(cY + centroid_width)] = 0
            final_mask += mask

        # Delete pixels in input depth
        modified_input_depth = input_depth
        modified_input_depth[final_mask == 255] = 0.0

        return modified_input_depth

    def _modify_depth_delete_random_pixels(self, input_depth, percentageHoles, resize_factor=1):
        """Randomly remove a percentage of pixels from the input depth image.

        Args:
            input_depth (numpy.ndarray): Depth Image. dtype=np.float32
            percentageHoles (float): The percentage of pixels from input depth to be randomly removed (made 0.0)
                                     Range of expected values: [0.0, 1.0]
            resize_factor (int): The number by which output image height and width will be divided before deleting
                                 pixels. This has the effect of creating larger holes. Note that output image size
                                 is defined upon init of this class.
                                 Eg: If the resize factor is 4, each deleted pixel will create a hole that is 4 pixels
                                 wide in height and width.

        Returns:
            numpy.ndarray: Depth image with holes.
        """
        # Create holes in depth
        if resize_factor < 1:
            raise ValueError('The resize_factor should be an int that is >= 1')

        tmp_input_depth = cv2.resize(
            input_depth, (int(self.outputImgWidth / resize_factor), int(self.outputImgHeight / resize_factor)),
            interpolation=cv2.INTER_NEAREST)

        flatten_array = tmp_input_depth.flatten()
        indices = np.random.choice(np.arange(flatten_array.size),
                                   replace=False,
                                   size=int(tmp_input_depth.size * percentageHoles))
        flatten_array[indices] = 0.0
        input_depth_holes = flatten_array.reshape(tmp_input_depth.shape)

        input_depth_holes = cv2.resize(input_depth_holes, (self.outputImgWidth, self.outputImgHeight),
                                       interpolation=cv2.INTER_NEAREST)

        return input_depth_holes

    def _modify_depth_delete_masks(self, rgb_image, input_depth, dilate_mask=True):
        # Run inference to get mask of transparent objects
        self.mask_predicted = self.inferenceMasks.runOnNumpyImage(rgb_image, dilate_mask=True)
        self.mask_predicted = cv2.resize(self.mask_predicted, (self.outputImgWidth, self.outputImgHeight),
                                         interpolation=cv2.INTER_NEAREST)
        # Mask depth of all transparent objects (detected depth of transparent objects is unreliable)
        modified_input_depth = input_depth.copy()
        modified_input_depth[self.mask_predicted > 0] = 0.0

        return modified_input_depth

    def _modify_depth_delete_masks_normals(self, rgb_image, input_depth):
        # SURFACE NORMALS
        surface_normals, surface_normals_rgb = self.inferenceNormals.runOnNumpyImage(rgb_image)

        # MASKS MODEL WAS TRAINED ON OUR REPRESENTATION OF SURFACE NORMALS - CONVERT NORMALS BACK TO OUR REPRESENTATION
        # Rotate Surface Normals into depth2depth notation
        normals_list = np.reshape(surface_normals, (3, -1)).transpose((1, 0))
        r = scipy.spatial.transform.Rotation.from_euler('x', 270, degrees=True)
        normals_list_rotated = r.apply(normals_list)
        surface_normals = np.reshape(normals_list_rotated.transpose(1, 0), surface_normals.shape)

        # Create RGB Viz of Normals
        surface_normals_rgb = self.inferenceNormals._normal_to_rgb(surface_normals.transpose((1, 2, 0)))

        # Run inference to get mask of transparent objects
        self.mask_predicted = self.inferenceMasks.runOnNumpyImage(surface_normals_rgb)
        self.mask_predicted = cv2.resize(self.mask_predicted, (self.outputImgWidth, self.outputImgHeight),
                                         interpolation=cv2.INTER_NEAREST)

        # Mask depth of all transparent objects (detected depth of transparent objects is unreliable)
        modified_input_depth = input_depth.copy()
        modified_input_depth[self.mask_predicted > 0] = 0.0

        return modified_input_depth

    def _complete_depth(self, input_image, input_depth, inertia_weight=1000, smoothness_weight=0.001, tangent_weight=1):
        """This function takes depth with holes, uses depth2depth alogorithm to output the completed depth image.
           Resizes the input rgb and depth image, creates depth with holes based on the percentage given.
           Runs inference on normals and outlines which along with depth with holes are the input for depth2depth
           algorithm. depth2depth algorithm: https://github.com/yindaz/DeepCompletionRelease
           Depth2depth is a C++ executable which requires the input files to saved to disk. We save the intermediary
           outputs to a tmp directory.

        Args:
            input_image (numpy.ndarray, uint8): Input rgb image. Shape=(H, W, 3)
            input_depth (numpy.ndarray, float32): Depth Image. Shape=(H, W)
            inertia_weight (float, optional): Defaults to 1000. The strength of the penalty on the difference between
                                              input and the output depth map on observed pixels. Set this value higher
                                              if you want to maintain the observed depth from input_depth.png.
            smoothness_weight (float, optional): Defaults to 0.001. The strength of the penalty on the difference
                                                 between the depths of neighboring pixels. Higher smoothness weight
                                                will produce soap-film-like result.
            tangent_weight (float, optional): Defaults to 1. The universal strength of the surface normal constraint.
                                              Higher tangent weight will force the output to have the same surface
                                              normal with the given one.

        Returns:
            numpy.ndarray: output_depth - completed output depth

        Raises:
            DepthCompletionError: depth2depth module encountered error during depth completion.
        """
        if ((input_image.shape[0], input_image.shape[1]) != input_depth.shape):
            raise ValueError('The height and width of input image and input depth should be the same')

        # Filter input depth (remove pixels close to camera)
        MIN_DIST_TO_CAMERA = 0.1  # meters
        input_depth[input_depth < MIN_DIST_TO_CAMERA] = 0.0

        # save input depth for depth2depth module
        scaled_input_depth = utils.scale_depth(input_depth)
        utils.save_uint16_png(self._PATH_TMP_INPUT_DEPTH, scaled_input_depth)

        # SURFACE NORMALS
        self.surface_normals, self.surface_normals_rgb = self.inferenceNormals.runOnNumpyImage(input_image)
        # resize surface normals
        self.surface_normals = self.surface_normals.transpose(1, 2, 0)
        self.surface_normals = cv2.resize(self.surface_normals, (self.outputImgWidth, self.outputImgHeight),
                                          interpolation=cv2.INTER_NEAREST)
        self.surface_normals = self.surface_normals.transpose(2, 0, 1)
        # resize surface normals rgb
        self.surface_normals_rgb = cv2.resize(self.surface_normals_rgb, (self.outputImgWidth, self.outputImgHeight),
                                              interpolation=cv2.INTER_NEAREST)

        # Write surface normals to file for depth2depth module
        # NOTE: The hdf5 expected shape is (3, height, width), float32
        # THE ORDER OF AXES IS DIFFERENT FOR DEPTH2DEPTH: (Note that RGB corresponds to positive XYZ axes)
        #   - Ours:        R: Right,  G: Up,               B: Towards Camera  (Y-up notation)
        #   - depth2depth: R: Right,  G: Away from Camera, B: Up              (Z-up notation)
        with h5py.File(self._PATH_TMP_SURFACE_NORMALS, "w") as f:
            dset = f.create_dataset('/result', data=self.surface_normals)

        # OUTLINES
        self.occlusion_weight, self.occlusion_weight_rgb, self.outlines_rgb = self.inferenceOutlines.runOnNumpyImage(
            input_image)
        # resize occlusion_weight occlusion_weight_rgb
        self.occlusion_weight = cv2.resize(self.occlusion_weight, (self.outputImgWidth, self.outputImgHeight),
                                           interpolation=cv2.INTER_NEAREST)
        self.occlusion_weight_rgb = cv2.resize(self.occlusion_weight_rgb, (self.outputImgWidth, self.outputImgHeight),
                                               interpolation=cv2.INTER_NEAREST)
        self.outlines_rgb = cv2.resize(self.outlines_rgb, (self.outputImgWidth, self.outputImgHeight),
                                       interpolation=cv2.INTER_NEAREST)

        # Write the output to file for depth2depth module
        utils.save_uint16_png(self._PATH_TMP_OCCLUSION_WEIGHTS, self.occlusion_weight)

        # DEPTH2DEPTH
        """This is what a depth2depth cmd looks like:
        ./depth2depth\
            "data/input-depth.png" \
            "data/output-depth.png" \
            -xres 512 -yres 288 \
            -fx 370 -fy 370 \+
            -cx 256 -cy 144 \
            -inertia_weight 1000 \
            -smoothness_weight 0.001 \
            -tangent_weight 1 \
            -input_normals "data/output-surface-normals.h5" \
            -input_tangent_weight "data/output-outlines.png"
        """
        cmd = [
            self.depth2depthExecutable,
            self._PATH_TMP_INPUT_DEPTH,
            self._PATH_TMP_OUTPUT_DEPTH,
            '-xres',
            str(self.outputImgWidth),
            '-yres',
            str(self.outputImgHeight),
            '-fx',
            str(self.fx),
            '-fy',
            str(self.fy),
            '-cx',
            str(self.cx),
            '-cy',
            str(self.cy),
            '-inertia_weight',
            str(inertia_weight),
            '-smoothness_weight',
            str(smoothness_weight),
            '-tangent_weight',
            str(tangent_weight),
            '-input_normals',
            self._PATH_TMP_SURFACE_NORMALS,
            '-input_tangent_weight',
            self._PATH_TMP_OCCLUSION_WEIGHTS,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        RESULT_SUCCESS = 255

        if (result.returncode != RESULT_SUCCESS):
            raise DepthCompletionError('Depth Completion Failed: ReturnCode: {}\n  '.format(result.returncode) +
                                       str(result.stdout.decode(sys.getfilesystemencoding(), errors='ignore')) +
                                       str(result.stderr.decode(sys.getfilesystemencoding(), errors='ignore')))

        # Read result Output Depth
        scaled_output_depth = cv2.imread(self._PATH_TMP_OUTPUT_DEPTH, cv2.IMREAD_UNCHANGED)
        output_depth = utils.unscale_depth(scaled_output_depth)  # output depth in meters

        return output_depth

    # TODO: Delete random_holes mode, it is no longer required.
    def depth_completion(self,
                         rgb_image,
                         input_depth,
                         inertia_weight=1000,
                         smoothness_weight=0.001,
                         tangent_weight=1,
                         mode_modify_input_depth=None,
                         dilate_mask=True):
        """Runs depth completion on an input RGB and depth image.
        It will resize and modify the input depth as per the mode chosen:
            - None: No modification to the input depth
            - random_holes: It will randomly delete a percentage of pixels in input depth
            - mask: It will mask out all the pixels belonging to transparent objects using a deep neural network, with
                    the RGB image as input.
            - mask_normals: Mask predicted from the surface normals as input.

        Args:
            rgb_image (numpy.ndarray): Input RGB image
            input_depth (numpy.ndarray): Input Depth image
            inertia_weight (float, optional): Defaults to 1000. The strength of the penalty on the difference between
                                              input and the output depth map on observed pixels. Set this value higher
                                              if you want to maintain the observed depth from input_depth.png.
            smoothness_weight (float, optional): Defaults to 0.001. The strength of the penalty on the difference
                                                 between the depths of neighboring pixels. Higher smoothness weight
                                                will produce soap-film-like result.
            tangent_weight (float, optional): Defaults to 1. The universal strength of the surface normal constraint.
                                              Higher tangent weight will force the output to have the same surface
                                              normal with the given one.
            mode_modify_input_depth (str, optional): What method to use to modify input depth (see desc above).
                                                     If using random_holes, then additional params of "percentageHoles"
                                                     and "resize_factor" need to be passed.
                                                     Defaults to None.
            dilate_mask (bool): Only used if mode is "mask". If True, will dilate the mask predicted by mask
                                prediction etwork.

        Raises:
            NotImplementedError: [description]

        Returns:
            numpy.ndarray: Output Depth
            numpy.ndarray: Filtered Output Depth
        """
        # Resize images to output image size
        self.input_image = cv2.resize(rgb_image, (self.outputImgWidth, self.outputImgHeight),
                                      interpolation=cv2.INTER_LINEAR)
        # Clean input depth
        input_depth[np.isnan(input_depth)] = 0
        input_depth[np.isinf(input_depth)] = 0
        self.orig_input_depth = cv2.resize(input_depth, (self.outputImgWidth, self.outputImgHeight),
                                           interpolation=cv2.INTER_NEAREST)
        # NOTE: Clipping the input depth results in the input depth getting distorted, hence depth completion results are affected.
        self.orig_input_depth[self.orig_input_depth < self.min_depth] = 0.0
        self.orig_input_depth[self.orig_input_depth > self.max_depth] = self.max_depth

        # modify input depth
        if mode_modify_input_depth is None or mode_modify_input_depth == '':
            self.input_depth = self.orig_input_depth
            self.mask_predicted = np.zeros((self.outputImgHeight, self.outputImgWidth), dtype=np.uint8)
        elif mode_modify_input_depth == 'mask':
            if not self.masksWeightsFile:
                raise ValueError('no "masksWeightsFile" has been specified')
            self.input_depth = self._modify_depth_delete_masks(self.input_image,
                                                               self.orig_input_depth,
                                                               dilate_mask=dilate_mask)
        else:
            raise NotImplementedError('Mode of depth modification can only be None, or "mask"')

        # run depth2depth
        self.output_depth = self._complete_depth(self.input_image,
                                                 self.input_depth,
                                                 inertia_weight=inertia_weight,
                                                 smoothness_weight=smoothness_weight,
                                                 tangent_weight=tangent_weight)

        # Filter Output Depth
        # See: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#bilateralfilter
        if self.d > 0:
            self.filtered_output_depth = cv2.bilateralFilter(self.output_depth,
                                                             d=self.d,
                                                             sigmaColor=self.sigmaColor,
                                                             sigmaSpace=self.sigmaSpace,
                                                             borderType=cv2.BORDER_REFLECT)
        else:
            self.filtered_output_depth = self.output_depth

        # Reset values that are meant later to be fed into compute errors
        self.mask_valid_region = np.zeros((self.outputImgHeight, self.outputImgWidth), dtype=np.uint8)
        self.depth_gt = np.zeros((self.outputImgHeight, self.outputImgWidth), dtype=np.float32)

        return self.output_depth, self.filtered_output_depth

    def depth_completion_synthetic_centroid(self,
                                            rgb_image,
                                            input_depth,
                                            json_file_path,
                                            variant_mask_path,
                                            centroid_width=5,
                                            inertia_weight=1000,
                                            smoothness_weight=0.001,
                                            tangent_weight=1):
        """Runs Depth Completion on Synthetic Images, where the objects are deleted using ground truth
        except for a centroid in the input depth image.

        Args:
            rgb_image (numpy.ndarray): Input RGB image
            input_depth (numpy.ndarray): Input Depth Image
            json_file_path (str): Path to json file corresponding to the input image
            variant_mask_path (ste): Path to the Variant Mask corresponding to the input image
            centroid_width (int, optional): Num of pixels left at the centroid of each object. Defaults to 5.
            inertia_weight (float, optional): See depth_completion(). Defaults to 1000.
            smoothness_weight (float, optional): See depth_completion(). Defaults to 0.001.
            tangent_weight (float, optional): See depth_completion(). Defaults to 1.

        Returns:
            numpy.ndarray: Output Depth
            numpy.ndarray: Filtered Output Depth
        """
        # Resize images to output image size
        self.input_image = cv2.resize(rgb_image, (self.outputImgWidth, self.outputImgHeight),
                                      interpolation=cv2.INTER_LINEAR)
        self.orig_input_depth = cv2.resize(input_depth, (self.outputImgWidth, self.outputImgHeight),
                                           interpolation=cv2.INTER_NEAREST)

        # modify input depth
        self.input_depth = _modify_synthetic_depth_delete_objects(self.orig_input_depth,
                                                                  json_file_path,
                                                                  variant_mask_path,
                                                                  centroid_width=centroid_width)

        # run depth2depth
        self.output_depth, self.filtered_output_depth = depth_completion(self.input_image,
                                                                         self.input_depth,
                                                                         inertia_weight=inertia_weight,
                                                                         smoothness_weight=smoothness_weight,
                                                                         tangent_weight=tangent_weight,
                                                                         mode_modify_input_depth=None)

        return self.output_depth, self.filtered_output_depth

    def compute_errors(self, gt, pred, mask):
        """Compute error for depth as required for paper (RMSE, REL, etc)
        Args:
            gt (numpy.ndarray): Ground truth depth (metric). Shape: [B, H, W], dtype: float32
            pred (numpy.ndarray): Predicted depth (metric). Shape: [B, H, W], dtype: float32
            mask (numpy.ndarray): Mask of pixels to consider while calculating error.
                                  Pixels not in mask are ignored and do not contribute to error.
                                  Shape: [B, H, W], dtype: bool

        Returns:
            dict: Various measures of error metrics
        """

        safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
        safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))

        self.mask_valid_region = mask
        self.depth_gt = gt

        gt = torch.from_numpy(gt)
        pred = torch.from_numpy(pred)
        mask = torch.from_numpy(mask).byte()

        gt = gt[mask]
        pred = pred[mask]
        thresh = torch.max(gt / pred, pred / gt)
        a1 = (thresh < 1.05).float().mean()
        a2 = (thresh < 1.10).float().mean()
        a3 = (thresh < 1.25).float().mean()

        rmse = ((gt - pred)**2).mean().sqrt()
        rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
        log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
        abs_rel = ((gt - pred).abs() / gt).mean()
        mae = (gt - pred).abs().mean()
        sq_rel = ((gt - pred)**2 / gt).mean()

        measures = {
            'a1': round(a1.item() * 100, 5),
            'a2': round(a2.item() * 100, 5),
            'a3': round(a3.item() * 100, 5),
            'rmse': round(rmse.item(), 5),
            'rmse_log': round(rmse_log.item(), 5),
            'log10': round(log10.item(), 5),
            'abs_rel': round(abs_rel.item(), 5),
            'sq_rel': round(sq_rel.item(), 5),
            'mae': round(mae.item(), 5),
        }
        return measures
