"""Script that loads in a single pointcloud and creates seq of jpg files within a given dire for turn-table like animation
"""
import argparse
import time
import os
import math
import shutil

import numpy as np
import open3d as o3d
import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a seq of images for animation of pointcloud. To rotate from -45 to 45 deg, give initial'
        ' rotation in y as -45 and a sweep of 90 deg. 1 image will be taken per deg of rotation')
    parser.add_argument('-p', '--path', type=str, help='path to pointcloud')
    parser.add_argument('-d', '--dest', type=str, help='path to output dir')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite dest directory without confirmation')
    parser.add_argument('-x', '--angle_x', type=float, default=155, help='Initial Angle (deg) to rotate in X (Pitch)')
    parser.add_argument('-y', '--angle_y', type=float, default=0, help='Initial Angle (deg) to rotate in y (Yaw)')
    parser.add_argument('-s', '--yaw_sweep', type=float, default=0, help='How many deg model will rotate (Yaw) to'
                        ' generate animation')
    args = parser.parse_args()

    print("Load a ply point cloud, print it, and render it")
    if not os.path.exists(args.path):
        print('Input ptcloud does not exist: {}'.format(args.path))
        exit()
    pcd = o3d.io.read_point_cloud(args.path)

    # Estimate Normals
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10000))
    # pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    # pcd.normalize_normals()
    # pcd_normals = np.asarray(pcd.normals)
    # pcd_normals[:, 1] = pcd_normals[:, 1] * -1.0  # Orient normals according to (Y-Up, X-Right)
    # pcd_normals[:, 2] = pcd_normals[:, 2] * -1.0  # Orient normals according to (Y-Up, X-Right)
    # rgb_points = (((pcd_normals + 1) / 2) * 255).astype(np.uint8)

    print('Applying initial Rotation to pointcloud for desired view')
    pcd.rotate(pcd.get_rotation_matrix_from_xyz(np.array([math.radians(args.angle_x), 0, 0], dtype=np.float64)))
    pcd.rotate(pcd.get_rotation_matrix_from_xyz(np.array([0, math.radians(args.angle_y), 0], dtype=np.float64)))

    output_dir = args.dest
    if not os.path.isdir(output_dir):
        print('Creating dir to store results: {}'.format(output_dir))
        os.makedirs(output_dir)
    else:
        if args.overwrite:
            cont = 'y'
        else:
            cont = input("Folder already exists! Overwrite? (Y/N): ")

        if cont == 'y' or cont == 'Y':
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            exit()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.light_on = False
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_color_option = o3d.visualization.PointColorOption.Normal

    print('Creating seq of animation files')
    for ii in range(int(args.yaw_sweep)):
        ctr = vis.get_view_control()
        DEG_TO_ROTATE_YAW = 1
        pcd.rotate(pcd.get_rotation_matrix_from_xyz(np.array([0, math.radians(DEG_TO_ROTATE_YAW), 0], dtype=np.float64)))

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(os.path.join(output_dir, "{:05d}.jpg".format(ii)))
    vis.destroy_window()
