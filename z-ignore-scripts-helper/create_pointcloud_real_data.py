"""Creates pointclouds from rgb and depth images of real data (has pairs for transparent and opaque objects)
Can create pointclouds colored either in original RGB or Surface Normals.
Surface Normals can either be estimated from point cloud, or passed in as a file
"""
import argparse
import os
import glob
import fnmatch
import struct
import yaml

import attrdict
import open3d as o3d
import numpy as np
import cv2
import imageio
import Imath
import OpenEXR
from PIL import Image

EXT_COLOR_IMG1 = '-transparent-rgb-img.jpg'
EXT_DEPTH_IMG1 = '-transparent-depth-img.exr'
EXT_PT1 = '-transparent-pt.ply'
EXT_COLOR_IMG2 = '-opaque-rgb-img.jpg'
EXT_DEPTH_IMG2 = '-opaque-depth-img.exr'
EXT_PT2 = '-opaque-pt.ply'
EXT_COLOR_IMG_R = '-rgb.jpg'
EXT_DEPTH_IMG_R = '-output-depth.exr'
EXT_PT_R = '-opaque-pt.ply'
EXT_NORMALS_RGB = '-normals-rgb.png'

CAM_INTRINSICS_FILENAME = 'camera_intrinsics.yaml'
RESULTS_SUBFOLDER = 'pointclouds'


def write_point_cloud(filename, xyz_points, rgb_points):
    """Creates and Writes a .ply point cloud file using RGB and Depth points derived from images.

    Args:
        filename (str): The path to the file which should be written. It should end with extension '.ply'
        xyz_points (numpy.ndarray): Shape=[-1, 3], dtype=np.float32
        rgb_points (numpy.ndarray): Shape=[-1, 3], dtype=np.uint8.
    """
    if xyz_points.dtype != 'float32':
        print('[ERROR]: xyz_points should be float32, it is {}'.format(xyz_points.dtype))
        exit()
    if rgb_points.dtype != 'uint8':
        print('[ERROR]: xyz_points should be uint8, it is {}'.format(rgb_points.type))
        exit()

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
                    struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2], rgb_points[i, 0].tostring(),
                                rgb_points[i, 1].tostring(), rgb_points[i, 2].tostring())))


def get_3d_points(color_image, depth_image, fx, fy, cx, cy):
    """Creates point cloud from rgb images and depth image

    Args:
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len (along x-axis, in pixels) of camera used to capture image.
        fy (int): The focal len (along y-axis, in pixels) of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    Returns:
        numpy.ndarray: camera_points - The XYZ location of each pixel. Shape: (num of pixels, 3)
        numpy.ndarray: color_points - The RGB color of each pixel. Shape: (num of pixels, 3)
    """
    # camera instrinsic parameters
    # camera_intrinsics  = [[fx 0  cx],
    #                       [0  fy cy],
    #                       [0  0  1]]
    camera_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.int32)

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    camera_points_x = np.multiply(pixel_x - camera_intrinsics[0, 2], (depth_image / camera_intrinsics[0, 0]))
    camera_points_y = np.multiply(pixel_y - camera_intrinsics[1, 2], (depth_image / camera_intrinsics[1, 1]))
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1, 3)

    color_points = color_image.reshape(-1, 3)

    # Note - Do not Remove invalid 3D points (where depth == 0), since it results in unstructured point cloud, which is not easy to work with using Open3D
    return camera_points, color_points


def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def create_pt_files(color_img_file, depth_img_file, filename, camera_params, radius=0.01, color_with_normals=False,
                    normals_file=None):
    """Create and save a pointcloud including Estimate Surface Normals and using it to color the pointcloud

    Args:
        color_img_file (ste): path to color image. Should have 9-digit prefix.
        depth_img_file (str): path to depth image. Should have 9-digit prefix.
        filename (str): The path to save output ptcloud
        radius (float, optional): Radius within which to search for nearest points for est of normals. Defaults to 0.01.
        color_with_normals (bool, optional): If true, will color the pointcloud with surface normals
        normals_file (str, optional): If given, will read surface normals from file, else will estimate normals from
                                      pointcloud.

    Raises:
        ValueError: If type is wrong
    """
    color_img = imageio.imread(color_img_file)
    depth_img = exr_loader(depth_img_file, ndim=1)
    depth_img[np.isnan(depth_img)] = 0.0
    depth_img[np.isinf(depth_img)] = 0.0

    HEIGHT = 144
    WIDTH = 256

    depth_img = cv2.resize(depth_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    color_img = cv2.resize(color_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    # ===================== SURFACE NORMAL ESTIMATION ===================== #
    height, width = color_img.shape[0], color_img.shape[1]
    fx = (float(width) / camera_params.xres) * camera_params.fx
    fy = (float(height) / camera_params.yres) * camera_params.fy
    cx = (float(width) / camera_params.xres) * camera_params.cx
    cy = (float(height) / camera_params.yres) * camera_params.cy

    # Create PointCloud
    camera_points, color_points = get_3d_points(color_img, depth_img, int(fx), int(fy), int(cx), int(cy))
    valid_depth_ind = np.where(depth_img.flatten() > 0)[0]
    xyz_points = camera_points[valid_depth_ind, :].astype(np.float32)
    rgb_points = color_points[valid_depth_ind, :]

    # Estimate Normals for coloring pointclouds
    if normals_file is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=10000))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        pcd.normalize_normals()
        pcd_normals_2 = np.asarray(pcd.normals).astype(np.float32)
        pcd_normals = pcd_normals_2.copy()
        pcd_normals[:, 1] = pcd_normals[:, 1] * -1.0  # Orient normals according to (Y-Up, X-Right)
        pcd_normals[:, 2] = pcd_normals[:, 2] * -1.0  # Orient normals according to (Y-Up, X-Right)
        normal_rgb_points = (((pcd_normals + 1) / 2) * 255).astype(np.uint8)
    else:
        normal_rgb = imageio.imread(normals_file)
        normal_rgb_points = normal_rgb.reshape(-1, 3)
        normal_rgb_points = normal_rgb_points[valid_depth_ind, :]

    # Save point cloud
    if color_with_normals is True:
        write_point_cloud(filename, xyz_points, normal_rgb_points)
    else:
        write_point_cloud(filename, xyz_points, rgb_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates pointclouds from rgb and depth images of real data')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    parser.add_argument('-n', '--num', required=True, type=int,
                        help='prefix of image to read. If -1, will process all images in source dir')
    parser.add_argument('-r',
                        '--radius',
                        default=0.01,
                        type=float,
                        help='Radius to search nearby points for normals estimation')
    parser.add_argument('-o',
                        '--overwrite',
                        action="store_true",
                        default=False,
                        help='If passed, will overwrite old predictions')
    parser.add_argument('-e',
                        '--est_normals',
                        action="store_true",
                        default=False,
                        help='If passed, will estimate normals using given radius rather than read from disk')
    parser.add_argument('-c',
                        '--color_with_normals',
                        action="store_true",
                        default=False,
                        help='If passed, will color pointcloud with estimated normals')
    
    args = parser.parse_args()

    # Check valid dir, files
    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()

    if not os.path.isdir(os.path.join(args.source_dir, RESULTS_SUBFOLDER)):
        os.mkdir(os.path.join(args.source_dir, RESULTS_SUBFOLDER))
        print('Created {} subfolder to store pointclouds'.format(RESULTS_SUBFOLDER))

    # Load Config File
    CONFIG_FILE_PATH = os.path.join(args.source_dir, CAM_INTRINSICS_FILENAME)
    if not os.path.isfile(CONFIG_FILE_PATH):
        print('\nError: Camera Intrinsics yaml does not exist: {}\n'.format(CONFIG_FILE_PATH))
        exit()
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    camera_params = attrdict.AttrDict(config_yaml)

    color_with_normals = args.color_with_normals

    if args.num > -1:
        color_img_file_t = os.path.join(args.source_dir, '{:09d}'.format(args.num) + EXT_COLOR_IMG1)
        depth_img_file_t = os.path.join(args.source_dir, '{:09d}'.format(args.num) + EXT_DEPTH_IMG1)
        color_img_file_o = os.path.join(args.source_dir, '{:09d}'.format(args.num) + EXT_COLOR_IMG2)
        depth_img_file_o = os.path.join(args.source_dir, '{:09d}'.format(args.num) + EXT_DEPTH_IMG2)

        prefix = os.path.basename(color_img_file_t)[:9]
        filename = os.path.join(args.source_dir, RESULTS_SUBFOLDER, prefix + EXT_PT1)
        create_pt_files(color_img_file_t, depth_img_file_t, filename, camera_params, radius=args.radius,
                        color_with_normals=color_with_normals)
        
        prefix = os.path.basename(color_img_file_o)[:9]
        filename = os.path.join(args.source_dir, RESULTS_SUBFOLDER, prefix + EXT_PT2)
        create_pt_files(color_img_file_o, depth_img_file_o, filename, camera_params, radius=args.radius,
                        color_with_normals=color_with_normals)
    else:
        color_img_list_t = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_COLOR_IMG1)))
        depth_img_list_t = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_IMG1)))
        color_img_list_o = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_COLOR_IMG2)))
        depth_img_list_o = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_IMG2)))

        if args.est_normals is True:
            normals_file_list = [None] * len(color_img_list_t)
        else:
            normals_file_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_NORMALS_RGB)))

        for (color_img_file_t, depth_img_file_t, normals_file) in zip(color_img_list_t, depth_img_list_t, normals_file_list):
            prefix = os.path.basename(color_img_file_t)[:9]
            filename = os.path.join(args.source_dir, RESULTS_SUBFOLDER, prefix + EXT_PT1)
            print('Estimating normals for transparent file {}'.format(prefix))
            if os.path.isfile(filename) and args.overwrite is False:
                continue

            create_pt_files(color_img_file_t, depth_img_file_t, filename, camera_params, radius=args.radius,
                            color_with_normals=color_with_normals, normals_file=normals_file)

        for (color_img_file_o, depth_img_file_o, normals_file) in zip(color_img_list_o, depth_img_list_o, normals_file_list):
            prefix = os.path.basename(color_img_file_o)[:9]
            filename = os.path.join(args.source_dir, RESULTS_SUBFOLDER, prefix + EXT_PT2)
            print('Estimating normals for opaque file {}'.format(prefix))
            if os.path.isfile(filename) and args.overwrite is False:
                continue

            create_pt_files(color_img_file_o, depth_img_file_o, filename, camera_params, radius=args.radius,
                            color_with_normals=color_with_normals, normals_file=normals_file)

