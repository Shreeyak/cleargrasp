"""Our real images dataset needs ground truth surface normals and outlines.
This script will calculate these and save it for training and eval.
"""
import argparse
import os
import glob
import fnmatch
import struct
import yaml

import open3d as o3d
import numpy as np
import attrdict
import cv2
import imageio
import Imath
import OpenEXR
from PIL import Image

EXT_COLOR_IMG = '-opaque-rgb-img.jpg'
EXT_DEPTH_IMG = '-opaque-depth-img.exr'
EXT_SURFACE_NORMAL = '-normals.exr'
EXT_SURFACE_NORMAL_RGB = '-normals-rgb.png'
EXT_MASK = '-mask.png'
EXT_OUTLINES = '-outlines.png'
EXT_OUTLINES_RGB = '-outlines-rgb.png'
CAM_INTRINSICS_FILENAME = 'camera_intrinsics.yaml'


def write_point_cloud(filename, xyz_points, rgb_points, normals_points):
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
    if normals_points.dtype != 'float32':
        print('[ERROR]: normals_points should be float32, it is {}'.format(normals_points.dtype))
        exit()

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property float nx\n', 'utf-8'))
        fid.write(bytes('property float ny\n', 'utf-8'))
        fid.write(bytes('property float nz\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(
                bytearray(
                    struct.pack("ffffffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2], normals_points[i, 0],
                                normals_points[i, 1], normals_points[i, 2], rgb_points[i, 0].tostring(),
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


def exr_saver(EXR_PATH, ndarr, ndim=3):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()
    elif ndim == 1:
        # Check params
        if len(ndarr.shape) != 2:
            raise ValueError(('The shape of the tensor should be (height x width) for ndim = 1. ' +
                              'Given shape is {}'.format(ndarr.shape)))

        # Convert each channel to strings
        Rs = ndarr[:, :].astype(np.float16).tostring()

        # Write the color channel to the output file
        HEADER = OpenEXR.Header(ndarr.shape[1], ndarr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "R"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs})
        out.close()


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


# def outlines_from_depth(depth_img_orig):
#     '''Create outlines from the depth image
#     This is used to create a binary mask of the occlusion boundaries, also referred to as outlines of depth.
#     Outlines refers to areas where there is a sudden large change in value of depth, i.e., the gradient is large.

#     Example: the borders of an object against the background will have large gradient as depth value changes from
#     the object to that of background.

#     Args:
#         depth_img_orig (numpy.ndarray): Shape (height, width), dtype=float32. The depth image where each pixel
#             contains the distance to the object in meters.

#     Returns:
#         numpy.ndarray: Shape (height, width), dtype=uint8: Outlines from depth image
#     '''
#     # Sobel Filter Params
#     # These params were chosen using trial and error.
#     # NOTE!!! The max value of sobel output increases exponentially with increase in kernel size.
#     # Print the min/max values of array below to get an idea of the range of values in Sobel output.
#     kernel_size = 7
#     threshold = 8.0

#     # Apply Sobel Filter
#     depth_img_blur = cv2.GaussianBlur(depth_img_orig, (5, 5), 0)
#     sobelx = cv2.Sobel(depth_img_blur, cv2.CV_32F, 1, 0, ksize=kernel_size)
#     sobely = cv2.Sobel(depth_img_blur, cv2.CV_32F, 0, 1, ksize=kernel_size)
#     sobelx = np.abs(sobelx)
#     sobely = np.abs(sobely)
#     # print('minx:', np.amin(sobelx), 'maxx:', np.amax(sobelx))
#     # print('miny:', np.amin(sobely), 'maxy:', np.amax(sobely))

#     # Create Boolean Mask
#     sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
#     sobelx_binary[sobelx >= threshold] = True

#     sobely_binary = np.full(sobely.shape, False, dtype=bool)
#     sobely_binary[sobely >= threshold] = True

#     sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

#     sobel_result = np.zeros_like(depth_img_orig, dtype=np.uint8)
#     sobel_result[sobel_binary] = 255

#     # Clean the mask
#     kernel = np.ones((3, 3), np.uint8)
#     # sobel_result = cv2.erode(sobel_result, kernel, iterations=1)
#     # sobel_result = cv2.dilate(sobel_result, kernel, iterations=1)

#     # Make all depth values greater than 2.5m as 0
#     # This is done because the gradient increases exponentially on far away objects. So pixels near the horizon
#     # will create a large zone of depth edges, but we don't want those. We want only edges of objects seen in the scene.
#     max_depth_to_object = 2.5
#     sobel_result[depth_img_orig > max_depth_to_object] = 0

#     return sobel_result


# def outlines_from_masks(variant_mask):
#     '''Create outlines from the depth image
#     This is used to create a binary mask of the occlusion boundaries, also referred to as outlines of depth.
#     Outlines refers to areas where there is a sudden large change in value of depth, i.e., the gradient is large.

#     Example: the borders of an object against the background will have large gradient as depth value changes from
#     the object to that of background.

#     Args:
#         variant_mask (numpy.ndarray): Shape (height, width), dtype=uint8. The mask where transparent objects present 
#             indicated by value of 255, else zero.

#     Returns:
#         numpy.ndarray: Shape (height, width), dtype=uint8: Outlines from mask
#     '''
#     # Sobel Filter Params
#     # These params were chosen using trial and error.
#     # NOTE!!! The max value of sobel output increases exponentially with increase in kernel size.
#     # Print the min/max values of array below to get an idea of the range of values in Sobel output.
#     kernel_size = 7
#     threshold = 7.0

#     # Apply Sobel Filter
#     sobelx = cv2.Sobel(variant_mask, cv2.CV_32F, 1, 0, ksize=kernel_size)
#     sobely = cv2.Sobel(variant_mask, cv2.CV_32F, 0, 1, ksize=kernel_size)
#     sobelx = np.abs(sobelx)
#     sobely = np.abs(sobely)
#     # print('minx:', np.amin(sobelx), 'maxx:', np.amax(sobelx))
#     # print('miny:', np.amin(sobely), 'maxy:', np.amax(sobely))

#     # Create Boolean Mask
#     sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
#     sobelx_binary[sobelx >= threshold] = True

#     sobely_binary = np.full(sobely.shape, False, dtype=bool)
#     sobely_binary[sobely >= threshold] = True

#     sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

#     sobel_result = np.zeros_like(variant_mask, dtype=np.uint8)
#     sobel_result[sobel_binary] = 255

#     outlines_mask = np.zeros(variant_mask.shape, dtype=np.uint8)
#     outlines_mask[sobel_result == 255] = 255

#     return outlines_mask



# def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=1.0):
#     '''Converts a floating point depth image to uint8 or uint16 image.
#     The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.

#     Args:
#         depth_img (numpy.float32): Depth image, value is depth in meters
#         dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
#         max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
#             considered to be 0.0.
#     Raises:
#         ValueError: If wrong dtype is given

#     Returns:
#         numpy.ndarray: Depth image scaled to given dtype
#     '''

#     if dtype != np.uint16 and dtype != np.uint8:
#         raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

#     # Clip depth image to given range
#     depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
#     depth_img = np.ma.clip(depth_img, min_depth, max_depth)

#     # Get min/max value of given datatype
#     type_info = np.iinfo(dtype)
#     min_val = type_info.min
#     max_val = type_info.max

#     # Scale the depth image to given datatype range
#     depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
#     depth_img = depth_img.astype(dtype)

#     depth_img = np.ma.filled(depth_img, fill_value=0)  # Convert back to normal numpy array from masked numpy array

#     return depth_img


# def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
#               dynamic_scaling=False):
#     '''Generates RGB representation of a depth image.
#     To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

#     Holes in the depth image (0.0) appear black in color.

#     Args:
#         depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
#         min_depth (float): Min depth to be considered
#         max_depth (float): Max depth to be considered
#         color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
#                           Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

#                           Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
#                           This mapping changes from version to version.
#         reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
#         dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
#                                 image, rather that the passed arguments.
#     Returns:
#         numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
#     '''
#     # Map depth image to Color Map
#     if dynamic_scaling:
#         depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8,
#                                                 min_depth=max(depth_img[depth_img > 0].min(), min_depth),    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
#                                                 max_depth=min(depth_img.max(), max_depth))
#     else:
#         depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

#     if reverse_scale is True:
#         depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
#         depth_img_scaled = 255 - depth_img_scaled
#         depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

#     depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
#     depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

#     # Make holes in input depth black:
#     depth_img_mapped[depth_img_scaled == 0, :] = 0

#     return depth_img_mapped


def main():
    parser = argparse.ArgumentParser(description='Calculate surface normals from depth images in folder')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    parser.add_argument('-r',
                        '--radius',
                        default=0.02,
                        type=float,
                        help='Radius to seach for nearby points in meters for normals estimation')
    parser.add_argument('-n',
                        '--max_nn',
                        default=10000,
                        type=int,
                        help='Max number of nearby points for normals estimation')
    args = parser.parse_args()

    # Check valid dir, files
    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()

    color_img_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_COLOR_IMG)))
    depth_img_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_IMG)))
    mask_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_MASK)))
    num_color_imgs = len(color_img_list)
    if num_color_imgs != len(depth_img_list):
        print('[ERROR]: Number of color images ({}) and depth images ({}) do not match'.format(
            num_color_imgs, len(depth_img_list)))
        exit()
    if num_color_imgs != len(mask_list):
        print('[ERROR]: Number of color images ({}) and masks ({}) do not match'.format(
            num_color_imgs, len(mask_list)))
        exit()
    if num_color_imgs < 1:
        print('[ERROR]: Detected {} files in source dir'.format(num_color_imgs))
        exit()
    print('[INFO]: Processing {} files in source dir'.format(num_color_imgs))

    # Load Config File
    CONFIG_FILE_PATH = os.path.join(args.source_dir, CAM_INTRINSICS_FILENAME)
    if not os.path.isfile(CONFIG_FILE_PATH):
        print('\nError: Camera Intrinsics yaml does not exist: {}\n'.format(CONFIG_FILE_PATH))
        exit()
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    camera_params = attrdict.AttrDict(config_yaml)

    # Iterate over all the files
    for (color_img_file, depth_img_file, mask_file) in zip(color_img_list, depth_img_list, mask_list):
        prefix = os.path.basename(color_img_file)[:9]
        print('Estimating normals for file {} / {:09d}'.format(prefix, num_color_imgs))

        color_img = imageio.imread(color_img_file)
        mask = imageio.imread(mask_file)
        mask = mask[..., 0]
        depth_img = exr_loader(depth_img_file, ndim=1)
        depth_img[np.isnan(depth_img)] = 0.0
        depth_img[np.isinf(depth_img)] = 0.0

        # ===================== SURFACE NORMAL ESTIMATION ===================== #
        height, width = color_img.shape[0], color_img.shape[1]
        fx = (float(width) / camera_params.xres) * camera_params.fx
        fy = (float(height) / camera_params.yres) * camera_params.fy
        cx = (float(width) / camera_params.xres) * camera_params.cx
        cy = (float(height) / camera_params.yres) * camera_params.cy

        # Estimate Surface Normals
        camera_points, color_points = get_3d_points(color_img, depth_img, int(fx), int(fy), int(cx), int(cy))

        # Remove invalid 3D points (where depth == 0)
        valid_depth_ind = np.where(depth_img.flatten() > 0)[0]
        xyz_points = camera_points[valid_depth_ind, :].astype(np.float32)
        rgb_points = color_points[valid_depth_ind, :]

        # Create PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_points)

        # Estimate Normals
        o3d.geometry.estimate_normals(pcd,
                                      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.max_nn))
        o3d.geometry.orient_normals_towards_camera_location(pcd, camera_location=np.array([0., 0., 0.]))
        pcd.normalize_normals()
        pcd_normals = np.asarray(pcd.normals)

        # Retrieve Structure Surface Normals
        surface_normal = np.ones((height * width, 3), dtype=np.float32) * -1.0
        surface_normal[valid_depth_ind, 0] = pcd_normals[:, 0]
        surface_normal[valid_depth_ind, 1] = pcd_normals[:, 1] * -1.0  # Orient normals according to (Y-Up, X-Right)
        surface_normal[valid_depth_ind, 2] = pcd_normals[:, 2] * -1.0  # Orient normals according to (Y-Up, X-Right)
        surface_normal = surface_normal.reshape((height, width, 3))
        surface_normal_rgb = (((surface_normal + 1) / 2) * 255).astype(np.uint8)

        # Save Surface Normals
        exr_saver(os.path.join(args.source_dir, 'data', prefix + EXT_SURFACE_NORMAL),
                  surface_normal.transpose(2, 0, 1),
                  ndim=3)
        imageio.imwrite(os.path.join(args.source_dir, 'data', prefix + EXT_SURFACE_NORMAL_RGB), surface_normal_rgb)

        # # Save point cloud
        # pcd.colors = o3d.utility.Vector3dVector(color_points.astype(np.float64) / 255)
        normal_rgb_points = surface_normal_rgb.reshape(-1, 3)[valid_depth_ind, :]
        write_point_cloud(os.path.join(args.source_dir, 'data', prefix + '.ply'), xyz_points, normal_rgb_points,
                          pcd_normals.astype(np.float32))



        # # ===================== OUTLINE ===================== #
        # depth_edges = outlines_from_depth(depth_img)
        # mask_edges = outlines_from_masks(mask)
        # combined_outlines = np.zeros((depth_edges.shape), dtype=np.uint8)

        # # Remove the depth outlines from mask outlines
        # kernel = np.ones((5, 5), np.uint8)
        # depth_edges_dilated = cv2.dilate((combined_outlines == 1).astype(np.uint8) * 255, kernel, iterations=2)
        # mask_edges[depth_edges_dilated > 0] = 0

        # print('combined_outlines:', combined_outlines.shape)
        # print('mask_edges:', mask_edges.shape)
        # combined_outlines[depth_edges > 0] = 1
        # combined_outlines[mask_edges > 0] = 2

        # combined_outlines_rgb = np.zeros((depth_edges.shape[0], depth_edges.shape[1], 3), dtype=np.uint8)
        # combined_outlines_rgb[..., 0][combined_outlines == 0] = 255
        # combined_outlines_rgb[..., 1][combined_outlines == 1] = 255
        # combined_outlines_rgb[..., 2][combined_outlines == 2] = 255

        # # imageio.imwrite(os.path.join(args.source_dir, 'data', prefix + EXT_OUTLINES), combined_outlines)
        # # imageio.imwrite(os.path.join(args.source_dir, 'data', prefix + EXT_OUTLINES_RGB), combined_outlines_rgb)

        # depth_img_rgb = depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
        #       dynamic_scaling=True)

        # collage = np.hstack((color_img, depth_img_rgb, combined_outlines_rgb, surface_normal_rgb))
        # imageio.imwrite(os.path.join(args.source_dir, 'data', prefix + '-collage.png'), collage)



if __name__ == "__main__":
    main()