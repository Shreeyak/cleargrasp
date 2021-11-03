import argparse
import os
import glob

import imageio
import OpenEXR
import cv2
import Imath
import numpy as np


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
        exr_arr[np.isnan(exr_arr)] = 0.0
        exr_arr[np.isinf(exr_arr)] = 0.0
        return exr_arr


def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=1.0):
    '''Converts a floating point depth image to uint8 or uint16 image.
    The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.

    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
            considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given

    Returns:
        numpy.ndarray: Depth image scaled to given dtype
    '''

    if dtype != np.uint16 and dtype != np.uint8:
        raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

    # Clip depth image to given range
    depth_img = np.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    return depth_img


def depth2rgb(depth_img,
              min_depth=0.0,
              max_depth=1.5,
              color_mode=cv2.COLORMAP_JET,
              reverse_scale=False,
              dynamic_scaling=False,
              eps=0.01):
    '''Generates RGB representation of a depth image.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

    Holes in the depth image (0.0) appear black in color.

    Args:
        depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
        eps (float): Small value sub from min depth so min depth values don't appear black in some color schemes.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''
    # Map depth image to Color Map
    if dynamic_scaling:
        depth_img_scaled = _normalize_depth_img(
            depth_img,
            dtype=np.uint8,
            min_depth=max(
                depth_img[depth_img > 0].min(),
                min_depth) - eps,  # Add a small epsilon so that min depth does not show up as black (invalid pixels)
            max_depth=min(depth_img.max(), max_depth))
    else:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
        depth_img_scaled = 255 - depth_img_scaled
        depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


def normal_to_rgb(normals_to_convert, output_dtype='uint8'):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) for a numpy image, or a range of (0,1) to represent PIL Image.

    The surface normals' axes are mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
        output_dtype (str): format of output, possibel values = ['float', 'uint8']
                            if 'float', range of output (0,1)
                            if 'uint8', range of output (0,255)
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    if output_dtype == 'uint8':
        camera_normal_rgb *= 255
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    elif output_dtype == 'float':
        pass
    else:
        raise NotImplementedError(
            'Possible values for "output_dtype" are only float and uint8. received value {}'.format(output_dtype))

    return camera_normal_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous numbered images in a dataset, move to new folder and process.')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    parser.add_argument('-m',
                        '--max_depth',
                        default=1.5,
                        type=float,
                        help='Path to source dir',
                        metavar='path/to/dataset')
    parser.add_argument('--mode',
                        help='Which depth images we\'re coloring. \
                              Possible modes: ["syn_dataset", "real_dataset", "output"] ')
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()

    possible_modes = ["syn_dataset", "real_dataset", "output"]
    if args.mode not in possible_modes:
        raise ValueError('Given mode ({}) not valid. Possible modes are: {}'.format(args.mode, possible_modes))

    if args.mode == "syn_dataset":
        # Convert images within a folder of synthetic images dataset.
        EXT_DEPTH_SYN = '-depth-rectified.exr'
        EXT_NORMALS_SYN = '-cameraNormals.exr'
        EXT_DEPTH_SYN_RGB = '-depth-rgb.png'
        EXT_NORMALS_SYN_RGB = '-normals-rgb.png'

        depth_files_syn = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_SYN)))
        normal_files_syn = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_NORMALS_SYN)))
        print('Found {} depth files, {} normals files'.format(len(depth_files_syn), len(normal_files_syn)))

        for depth_file in depth_files_syn:
            depth_img = exr_loader(depth_file, ndim=1)
            depthrgb = depth2rgb(depth_img, min_depth=-0.05, max_depth=1.5, dynamic_scaling=True, eps=0.01)

            prefix_d = os.path.basename(depth_file)[:9]
            depthrgb_file = os.path.join(args.source_dir, prefix_d + EXT_DEPTH_SYN_RGB)
            imageio.imwrite(depthrgb_file, depthrgb)
            print('Converted depth file: {}'.format(prefix_d))

        for normal_file in normal_files_syn:
            normal_img = exr_loader(normal_file, ndim=3)
            normalrgb = normal_to_rgb(normal_img.transpose(1, 2, 0), output_dtype='uint8')

            prefix_n = os.path.basename(normal_file)[:9]
            normalrgb_file = os.path.join(args.source_dir, prefix_n + EXT_NORMALS_SYN_RGB)
            imageio.imwrite(normalrgb_file, normalrgb)
            print('Converted normals file {}'.format(prefix_n))

    elif args.mode == "real_dataset":
        # Convert images within a folder of real images. Should contain transparent/opaque rgb, depth, etc in root dir.
        EXT_DEPTH_T = '-transparent-depth-img.exr'
        EXT_DEPTH_O = '-opaque-depth-img.exr'
        EXT_DEPTH_RGB_T = '-transparent-depth-rgb.png'
        EXT_DEPTH_RGB_O = '-opaque-depth-rgb.png'
        EXT_DEPTH_R = '-output-depth.exr'
        EXT_DEPTH_RGB_R = '-output-depth-rgb.png'
        depth_files_t = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_T)))
        depth_files_o = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_O)))
        depth_files_r = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_DEPTH_R)))
        if len(depth_files_r) == 0:
            depth_files_r = [None] * len(depth_files_t)

        for (depth_file_t, depth_file_o, depth_file_r) in zip(depth_files_t, depth_files_o, depth_files_r):
            prefix_t = os.path.basename(depth_file_t)[:9]
            prefix_o = os.path.basename(depth_file_o)[:9]
            if depth_file_r is not None:
                prefix_r = os.path.basename(depth_file_r)[:9]
            else:
                prefix_r = prefix_t
            if prefix_t != prefix_o != prefix_r:
                raise ValueError

            depth_img_t = exr_loader(depth_file_t, ndim=1)
            depth_img_o = exr_loader(depth_file_o, ndim=1)
            depth_img_t[depth_img_t > args.max_depth] = 0
            depth_img_o[depth_img_o > args.max_depth] = 0

            min_depth = min(depth_img_t[depth_img_t > 0].min(), depth_img_o[depth_img_o > 0].min())
            max_depth = max(depth_img_t.max(), depth_img_o.max())
            eps = 1e-2

            depthrgb_t_file = os.path.join(args.source_dir, prefix_t + EXT_DEPTH_RGB_T)
            if not os.path.exists(depthrgb_t_file):
                depthrgb_t = depth2rgb(depth_img_t, min_depth - eps, max_depth)
                imageio.imwrite(depthrgb_t_file, depthrgb_t)

            depthrgb_o_file = os.path.join(args.source_dir, prefix_o + EXT_DEPTH_RGB_O)
            if not os.path.exists(depthrgb_o_file):
                depthrgb_o = depth2rgb(depth_img_o, min_depth - eps, max_depth)
                imageio.imwrite(depthrgb_o_file, depthrgb_o)

            # If output depth is present in folder, convert it, else leave it.
            depthrgb_r_file = os.path.join(args.source_dir, prefix_r + EXT_DEPTH_RGB_R)
            if depth_file_r is not None:
                if not os.path.exists(depthrgb_r_file):
                    depth_img_r = exr_loader(depth_file_r, ndim=1)
                    depthrgb_r = depth2rgb(depth_img_r, min_depth - eps, max_depth)
                    imageio.imwrite(depthrgb_r_file, depthrgb_r)
            else:
                pass
            print('Finished converting file: {}'.format(prefix_t))

    elif args.mode == "output":
        # Convert within output folder of eval
        SUBDIR_GT = 'gt-depth'
        SUBDIR_INPUT = 'input-depth'
        SUBDIR_OUTPUT = 'output-depth'

        f_list_depth_gt = sorted(glob.glob(os.path.join(args.source_dir, SUBDIR_GT, '*.exr')))
        f_list_depth_in = sorted(glob.glob(os.path.join(args.source_dir, SUBDIR_INPUT, '*.exr')))
        f_list_depth_out = sorted(glob.glob(os.path.join(args.source_dir, SUBDIR_OUTPUT, '*.exr')))
        if len(f_list_depth_gt) != len(f_list_depth_in) != len(f_list_depth_out):
            raise ValueError('Number of files not equal. GT: {}, IN: {}, OUT: {}'.format(
                len(f_list_depth_gt), len(f_list_depth_in), len(f_list_depth_out)
            ))

        for (f_depth_gt, f_depth_in, f_depth_out) in zip(f_list_depth_gt, f_list_depth_in, f_list_depth_out):
            depth_gt = exr_loader(f_depth_gt, ndim=1)
            depth_in = exr_loader(f_depth_in, ndim=1)
            depth_out = exr_loader(f_depth_out, ndim=1)
            depth_gt[depth_gt > args.max_depth] = 0
            depth_in[depth_in > args.max_depth] = 0
            depth_out[depth_out > args.max_depth] = 0

            min_depth = min(depth_gt[depth_gt > 0].min(), depth_in[depth_in > 0].min(), depth_out[depth_out > 0].min())
            max_depth = max(depth_gt.max(), depth_in.max(), depth_out.max())
            eps = 1e-2

            f_depth_gt_rgb = f_depth_gt.replace('.exr', '-rgb.png')
            f_depth_in_rgb = f_depth_in.replace('.exr', '-rgb.png')
            f_depth_out_rgb = f_depth_out.replace('.exr', '-rgb-without-edge-weights.png')
            depth_gt_rgb = depth2rgb(depth_gt, min_depth - eps, max_depth)
            depth_in_rgb = depth2rgb(depth_in, min_depth - eps, max_depth)
            depth_out_rgb = depth2rgb(depth_out, min_depth - eps, max_depth)

            imageio.imwrite(f_depth_gt_rgb, depth_gt_rgb)
            imageio.imwrite(f_depth_in_rgb, depth_in_rgb)
            imageio.imwrite(f_depth_out_rgb, depth_out_rgb)

            print('Finished converting file: {}'.format(os.path.basename(f_depth_gt_rgb)[:9]))


