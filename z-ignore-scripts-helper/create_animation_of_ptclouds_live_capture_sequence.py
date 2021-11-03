"""Script that loads in a seq of RGBD files for animation of input/output ptclouds.
Expects RGB images in .jpg format in a folder called 'color' and Depth images in .png format in folder called 'depth'.
Depth images will contain depth scaled by some factor.
Path should also contain a 'camera_intrinsic.json' file. This file will contain the depth scaling factor.
"""
import argparse
import time
import os
import json
import glob

import numpy as np
import open3d as o3d
import tqdm
import imageio
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a seq of images for animation of pointcloud')
    parser.add_argument('-p', '--path', type=str, help='path to pointcloud')
    parser.add_argument('-x', '--angle_x', type=float, default=2.7, help='Initial Angle to rotate in X')
    parser.add_argument('-y', '--angle_y', type=float, default=-1.1, help='Initial Angle to rotate in y')  # -0.785
    parser.add_argument('-m', '--max_depth', type=float, default=-1.2, help='And depth beyond this value is clipped')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print('Input dir does not exist: {}'.format(args.path))
        exit()

    PTCLOUD_SUBDIR = 'ptcloud'

    output_dir = os.path.join(args.path, PTCLOUD_SUBDIR)
    if not os.path.isdir(output_dir):
        print('Creating dir to store results: {}'.format(output_dir))
        os.makedirs(output_dir)
    else:
        print("[WARN]: Folder {} already exists! Overwriting".format(output_dir))

    COLOR_SUBDIR = 'color'
    DEPTH_SUBDIR = 'depth'
    INTRINSICS_FILE = 'camera_intrinsic.json'

    json_file = open(os.path.join(args.path, INTRINSICS_FILE), 'r')
    data = json.load(json_file)
    depth_scale = float(data["depth_scale"])
    width = int(data["width"])
    height = int(data["height"])
    intrinsic_matrix = data["intrinsic_matrix"]
    fx = int(intrinsic_matrix[0])
    fy = int(intrinsic_matrix[4])
    cx = int(intrinsic_matrix[6])
    cy = int(intrinsic_matrix[7])

    # print('data:', depth_scale, height, width, fx, fy, cx, cy)

    # Get list of indexes
    list_color_imgs = sorted(glob.glob(os.path.join(args.path, COLOR_SUBDIR, '*')))
    list_depth_imgs = sorted(glob.glob(os.path.join(args.path, DEPTH_SUBDIR, '*')))
    if len(list_color_imgs) != len(list_depth_imgs):
        raise ValueError('Num of color imgs {} and depth imgs {} not equal.'.format(len(list_color_imgs),
                                                                                    len(list_depth_imgs)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=720)
    # vis.get_render_option().load_from_json("data/renderoption.json")
    opt = vis.get_render_option()
    opt.light_on = False
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_color_option = o3d.visualization.PointColorOption.Normal
    opt.point_size = 3

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1.8)
    ctr.set_constant_z_near(0.05)

    # Read in Images
    for ii, sample_batched in enumerate(zip(list_color_imgs, list_depth_imgs)):
        path_color, path_depth = sample_batched

        color_img = imageio.imread(path_color)

        depth_img = o3d.io.read_image(path_depth)
        intr = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, intr, depth_scale=1000.0, depth_trunc=1500.0)

        # Downsample PtCloud
        pcd = pcd.voxel_down_sample(voxel_size=0.002)

        # Estimate Normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=20))
        pcd.normalize_normals()
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        # Rotate Ptcloud
        pcd.rotate(np.array([[args.angle_x], [0], [0]], dtype=np.float64))
        pcd.rotate(np.array([[0], [args.angle_y], [0]], dtype=np.float64))



        # transform_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -2.0, 1]])
        # r = R.from_euler('zyx', [0, 0, 30], degrees=True)
        # rot_mat = r.as_dcm()
        # transform_mat[:3, :3] = rot_mat
        # cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param.extrinsic = transform_mat
        # ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.add_geometry(pcd)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(os.path.join(output_dir, "{:09d}_ptcloud.png".format(ii)))
        vis.remove_geometry(pcd)

    vis.destroy_window()
