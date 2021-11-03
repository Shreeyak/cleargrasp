"""Script that loads in a list of pointclouds and creates seq of jpg files for turn-table like animation
of ground truth, output, yinda-output, densedepth
"""
import argparse
import concurrent.futures
import os
import glob
import itertools

import numpy as np
import open3d as o3d
import tqdm


SUBDIR_OUTPUT = 'output-point-cloud'
SUBDIR_GT = 'gt-point-cloud'
SUBDIR_IN = 'input-point-cloud'

# Selected Imgs
# IMG_LIST_VAL = [130, 123, 112, 103, 102, 99, 98, 96, 95, 94, 90, 89, 85, 84, 83, 82, 80, 79, 77, 76, 75, 74, 73, 71, 70, 69, 60, 52, 50, 49, 47, 45, 42, 41, 39, 38, 36, 31, 30, 28, 27, 25, 24, 22, 21, 10, 8, 6, 5, 4, 3, 2, 1, 0]
# IMG_LIST_VAL_NOISY = [153, 156, 157, 158, 159, 160, 161, 163, 165, 166, 172]
# IMG_LIST_TEST_D415 = [5, 7, 8, 9, 10, 11, 12, 13, 14, 19, 22, 27, 30, 34, 35, 38, 39, 40, 41, 42, 49, 50, 51, 52, 53, 55, 60, 65, 67, 69, 71, 74, 75, 79, 80, 82, 83, 85, 86, 87, 88, 89]
IMG_LIST_VAL_ABLATION = [28, 30, 69, 76]
# IMG_LIST_TEST_D415_ABLATION = [49, 60, 80, 89]

IMG_LIST = IMG_LIST_VAL_ABLATION


def create_seq_images_from_ptcloud(path_ptcloud, dir_output, angle_x=155, angle_y=-90):
    """
    Creates a ptcloud and saves out a sequence of images by rotating the ptcloud 180 deg around y axis

    Args:
        path_ptcloud (str): Path to ptcloud to animate
        dir_output (str): Path to dir to store images of seq
        angle_x (float, optional): Default=155 deg. The angle to rotate ptcloud initially in x-axis for visual appeal
        angle_y (float, optional): Default=-90 deg. The angle to rotate ptcloud initially in y-axis for visual appeal

    Returns:

    """
    pcd = o3d.io.read_point_cloud(path_ptcloud)

    # Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    pcd = pcd.select_down_sample(ind)

    # Estimate Normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    pcd.normalize_normals()

    # Rotate PtCloud for optimal viewing angle
    pcd.rotate(np.array([[np.deg2rad(angle_x)], [0], [0]], dtype=np.float64))
    pcd.rotate(np.array([[0], [np.deg2rad(angle_y)], [0]], dtype=np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=dir_output, width=1280, height=720, left=0, top=0)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.light_on = False
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_color_option = o3d.visualization.PointColorOption.Normal
    opt.point_size = 2

    DEG_TO_ROTATE_PER_FRAME = 1
    for iii in range(0, int(abs(angle_y) * 4 / DEG_TO_ROTATE_PER_FRAME)):
        if iii <= int(abs(angle_y) * 2):
            pcd.rotate(np.array([[0], [np.deg2rad(DEG_TO_ROTATE_PER_FRAME)], [0]], dtype=np.float64))
        else:
            pcd.rotate(np.array([[0], [np.deg2rad(DEG_TO_ROTATE_PER_FRAME) * -1], [0]], dtype=np.float64))

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(os.path.join(dir_output, "{:09d}.jpg".format(iii)))
    vis.destroy_window()

    return


def create_seq_for_image_num(dir_dest, image_num, _f_in, _f_gt, _f_ours, _f_yinda, _f_dense):
    dir_image_num = os.path.join(dir_dest, image_num)
    dir_in = os.path.join(dir_image_num, 'input')
    dir_gt = os.path.join(dir_image_num, 'gt')
    dir_ours = os.path.join(dir_image_num, 'ours')
    dir_yinda = os.path.join(dir_image_num, 'yinda')
    dir_dense = os.path.join(dir_image_num, 'dense')
    if not os.path.exists(dir_image_num):
        os.makedirs(dir_in)
        os.makedirs(dir_gt)
        os.makedirs(dir_ours)
        os.makedirs(dir_yinda)
        os.makedirs(dir_dense)
    else:
        print("[WARN]: Folder exists, overwriting: {}".format(dir_image_num))

    create_seq_images_from_ptcloud(_f_in, dir_in, angle_x=155, angle_y=-60)
    create_seq_images_from_ptcloud(_f_gt, dir_gt, angle_x=155, angle_y=-60)
    create_seq_images_from_ptcloud(_f_ours, dir_ours, angle_x=155, angle_y=-60)
    create_seq_images_from_ptcloud(_f_yinda, dir_yinda, angle_x=155, angle_y=-60)
    create_seq_images_from_ptcloud(_f_dense, dir_dense, angle_x=155, angle_y=-60)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a seq of images for animation of pointcloud')
    parser.add_argument('--dir_ours', type=str, help='path to folder of outputs (ours)')
    parser.add_argument('--dir_yinda', type=str, help='path to folder of outputs (yinda)')
    parser.add_argument('--dir_densedepth', type=str, help='path to folder of outputs (densedepth)')
    parser.add_argument('--dir_dest', type=str, help='path to output folder')
    args = parser.parse_args()

    dir_list = [args.dir_ours, args.dir_yinda, args.dir_densedepth]
    for _dir in dir_list:
        if not os.path.exists(_dir):
            print('Dir does not exist: {}'.format(_dir))
            exit()

    if not os.path.exists(args.dir_dest):
        os.makedirs(args.dir_dest, exist_ok=True)

    pt_list_in = sorted(glob.glob(os.path.join(args.dir_ours, SUBDIR_IN, '*.ply')))
    pt_list_gt = sorted(glob.glob(os.path.join(args.dir_ours, SUBDIR_GT, '*.ply')))
    pt_list_ours = sorted(glob.glob(os.path.join(args.dir_ours, SUBDIR_OUTPUT, '*.ply')))
    pt_list_yinda = sorted(glob.glob(os.path.join(args.dir_yinda, SUBDIR_OUTPUT, '*.ply')))
    pt_list_dense = sorted(glob.glob(os.path.join(args.dir_densedepth, SUBDIR_OUTPUT, '*.ply')))

    img_list = ['{:09d}'.format(prefix) for prefix in sorted(IMG_LIST)]
    _f_in = [s for s in pt_list_in if os.path.basename(s)[:9] in img_list]
    _f_gt = [s for s in pt_list_gt if os.path.basename(s)[:9] in img_list]
    _f_ours = [s for s in pt_list_ours if os.path.basename(s)[:9] in img_list]
    _f_yinda = [s for s in pt_list_yinda if os.path.basename(s)[:9] in img_list]
    _f_dense = [s for s in pt_list_dense if os.path.basename(s)[:9] in img_list]


    for ii in tqdm.tqdm(range(len(img_list))):
        create_seq_for_image_num(args.dir_dest,
                                 img_list[ii],
                                 _f_in[ii],
                                 _f_gt[ii],
                                 _f_ours[ii],
                                 _f_yinda[ii],
                                 _f_dense[ii])
