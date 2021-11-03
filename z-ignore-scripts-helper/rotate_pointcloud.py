'''This script aims to find those pixels that are below a certain height by deriving a
point cloud from depth image and transforming it to world co-ords using camera
transformation matrix.
'''
import json
import sys

import open3d as o3d
import numpy as np
import imageio
from scipy.spatial.transform import Rotation as R

sys.path.append('..')
from api import utils as api_utils

# Set pretty-print options for viewing values of transformation matrix
np.set_printoptions(precision=4, suppress=True)

depth_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/square-clear-plastic-bottle/sources/square-clear-plastic-bottle-lying-flat-val/source-files/depth-imgs-rectified/000000003-depth-rectified.exr'
rgb_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/square-clear-plastic-bottle/sources/square-clear-plastic-bottle-lying-flat-val/source-files/rgb-imgs/000000003-rgb.jpg'
json_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/square-clear-plastic-bottle/sources/square-clear-plastic-bottle-lying-flat-val/source-files/json-files/000000003-masks.json'
mask_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/square-clear-plastic-bottle/sources/square-clear-plastic-bottle-lying-flat-val/source-files/segmentation-masks/000000003-segmentation-mask.png'
pt_file_orig = 'pt_orig.ply'
pt_file_rot = 'pt_rot.ply'

CLIPPING_HEIGHT = 0.04  # Meters
IMG_HEIGHT = 1080
IMG_WIDTH = 1920

# Get Rotation Matrix and Euler Angles
json_f = open(json_file)
data = json.load(json_f)

rot_mat_json = data['camera']['world_pose']['matrix_4x4']
transform_mat = np.zeros((4, 4), dtype=np.float64)
transform_mat[0, :] = rot_mat_json[0]
transform_mat[1, :] = rot_mat_json[1]
transform_mat[2, :] = rot_mat_json[2]
transform_mat[3, :] = rot_mat_json[3]
# rot_mat = transform_mat[:3, :3]  # Note that the transformation matrix may be incorrect. Quaternions behaving properly.
translate_mat = transform_mat[:3, 3]
translate_mat = np.expand_dims(translate_mat, 1)

# Get Rot from Quaternions
quat_mat_json = data['camera']['world_pose']['rotation']['quaternion']
r = R.from_quat(quat_mat_json)
rot_euler_xyz = r.as_euler('xyz', degrees=False)
rot_euler_xyz = np.expand_dims(rot_euler_xyz, 1)

# Get PointCloud
depth_img = api_utils.exr_loader(depth_file, ndim=1)
rgb_img = imageio.imread(rgb_file)
fx = 1386
fy = 1386
cx = 960
cy = 540
xyz_points, rgb_points = api_utils._get_point_cloud(rgb_img, depth_img, fx, fy, cx, cy)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_points)
pcd.colors = o3d.utility.Vector3dVector((rgb_points / 255.0).astype(np.float64))
o3d.io.write_point_cloud(pt_file_orig, pcd)

# Rotate & Translate PointCloud To World Co-Ords
pcd.rotate(rot_euler_xyz, center=False, type=o3d.RotationType.ZYX)
pcd.translate(-1 * translate_mat)

# Color Low Pixels in PointCloud
rot_xyz_points = np.asarray(pcd.points)
rot_xyz_points = rot_xyz_points.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
z_plane = rot_xyz_points[:, :, 2] > (-1 * CLIPPING_HEIGHT)

rot_rgb_points = np.asarray(pcd.colors)
rot_rgb_points = rot_rgb_points.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
rot_rgb_points[z_plane, :] = [0.6, 0.8, 1.0]  # Make the pixels Red
rot_rgb_points = rot_rgb_points.reshape(-1, 3)
pcd.colors = o3d.utility.Vector3dVector(rot_rgb_points)
o3d.io.write_point_cloud(pt_file_rot, pcd)

# Color Low Pixels in Segmentation Mask
low_pixels_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
mask = imageio.imread(mask_file)
low_pixels_mask[mask == 255, 1] = 255
low_pixels_mask[z_plane, 0] = 255
imageio.imwrite('low_pixels_mask.png', low_pixels_mask)
