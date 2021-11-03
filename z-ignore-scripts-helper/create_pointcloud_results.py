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


def create_pt_files(color_img_file, depth_img_file, filename, camera_params):
    """Create and save a pointcloud including Estimate Surface Normals and using it to color the pointcloud

    Args:
        color_img_file (ste): path to color image. Should have 9-digit prefix.
        depth_img_file (str): path to depth image. Should have 9-digit prefix.
        filename (str): The path to save output ptcloud

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
    write_point_cloud(filename, xyz_points, rgb_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate surface normals from depth images in folder')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    parser.add_argument('-d',
                        '--dest_dir',
                        required=True,
                        type=str,
                        help='Path to dest dir to save files',
                        metavar='path/to/dataset')
    parser.add_argument('-i',
                        '--img_dir',
                        required=True,
                        type=str,
                        help='Path to dir of images to RGB viz of surface normal',
                        metavar='path/to/dataset')
    parser.add_argument('-c',
                        '--camera_intrinsics_file',
                        required=True,
                        type=str,
                        help='Path to yaml file of camera intrinsics',
                        metavar='path/to/dataset')
    parser.add_argument('-n', '--num_start', required=True, type=int, help='prefix of image to read')
    parser.add_argument('-m', '--num_end', required=True, type=int, help='prefix of image to read')
    args = parser.parse_args()

    EXT_COLOR_IMG = '-normal-rgb.png'
    EXT_DEPTH_IMG = '-output-depth.exr'
    EXT_PT = '-output-pt.ply'

    # Check valid dir, files
    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()
    if not os.path.isdir(args.dest_dir):
        print('\nError: Dest dir does not exist: {}\n'.format(args.dest_dir))
        exit()
    if not os.path.isdir(args.img_dir):
        print('\nError: Image dir does not exist: {}\n'.format(args.img_dir))
        exit()

    # Load Config File
    CONFIG_FILE_PATH = args.camera_intrinsics_file
    if not os.path.isfile(CONFIG_FILE_PATH):
        print('\nError: Camera Intrinsics yaml does not exist: {}\n'.format(CONFIG_FILE_PATH))
        exit()
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    camera_params = attrdict.AttrDict(config_yaml)


    for ii in range(args.num_start, args.num_end):
        color_img_file = os.path.join(args.img_dir, '{:09d}'.format(ii) + EXT_COLOR_IMG)
        depth_img_file = os.path.join(args.source_dir, '{:09d}'.format(ii) + EXT_DEPTH_IMG)
        output_pt_file = os.path.join(args.dest_dir, '{:09d}'.format(ii) + EXT_PT)
        create_pt_files(color_img_file, depth_img_file, output_pt_file, camera_params)

        print('Created ptcloud: {:09d}'.format(ii))