import argparse
import os
import cv2
import sys
import numpy as np
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils


INPUT_DEPTH_FILENAME = 'input-depth.png'
OUTPUT_DEPTH_FILENAME = 'output-depth.png'
EXPECTED_OUTPUT_DEPTH_FILENAME = 'expected-output-depth.png'
NORMALS_FILENAME = 'normals.h5'
OCCLUSION_WEIGHTS_FILENAME = 'occlusion-weight.png'

RGB_INPUT_DEPTH_FILENAME = 'input-depth-rgb.jpg'
RGB_OUTPUT_DEPTH_FILENAME = 'output-depth-rgb.jpg'
RGB_EXPECTED_OUTPUT_DEPTH_FILENAME = 'expected-output-depth-rgb.jpg'
RGB_NORMALS_FILENAME = 'normals-rgb.jpg'
RGB_OCCLUSION_WEIGHTS_FILENAME = 'occlusion-weight-rgb.jpg'

def scaled_depth_to_rgb_depth(sample_files_dir, input_filename, output_filename):
    path_scaled_depth_img = os.path.join(sample_files_dir, input_filename)
    if not os.path.isfile(path_scaled_depth_img):
        print('\nError: Source file does not exist: {}\n'.format(path_scaled_depth_img))
        exit()

    scaled_depth = cv2.imread(path_scaled_depth_img, cv2.IMREAD_UNCHANGED)
    metric_depth = utils.unscale_depth(scaled_depth)

    rgb_depth = utils.depth2rgb(metric_depth, dynamic_scaling=True, color_mode=cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(sample_files_dir, output_filename), rgb_depth)
    return


def normals_to_rgb_normals(sample_files_dir, input_filename, output_filename):
    path_input_normals = os.path.join(sample_files_dir, input_filename)
    if not os.path.isfile(path_input_normals):
        print('\nError: Source file does not exist: {}\n'.format(path_input_normals))
        exit()

    with h5py.File(path_input_normals, 'r') as hf:
        normals = hf.get('/result')
        normals = np.array(normals)
        hf.close()

    # From (3, height, width) to (height, width, 3)
    normals = normals.transpose(1, 2, 0)

    rgb_normals = utils.normal_to_rgb(normals, output_dtype='uint8')
    cv2.imwrite(os.path.join(sample_files_dir, output_filename), rgb_normals)
    return


def occlusion_to_rgb_occlusion(sample_files_dir, input_filename, output_filename):
    path_input = os.path.join(sample_files_dir, input_filename)
    if not os.path.isfile(path_input):
        print('\nError: Source file does not exist: {}\n'.format(path_input))
        exit()

    occlusion_boundary_weight = cv2.imread(path_input, cv2.IMREAD_UNCHANGED)
    SCALING_FACTOR_MUL = 1000
    NORM_FACTOR_UINT8 = SCALING_FACTOR_MUL / 255.0
    occlusion_boundary_weight_rgb = (occlusion_boundary_weight / NORM_FACTOR_UINT8).astype(np.uint8)
    occlusion_boundary_weight_rgb = cv2.applyColorMap(occlusion_boundary_weight_rgb, cv2.COLORMAP_OCEAN)
    cv2.imwrite(os.path.join(sample_files_dir, output_filename), occlusion_boundary_weight_rgb)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert the black, 16-bit .png depth images to RGB depth images.')

    parser.add_argument('-s',
                        '--sample_files_dir',
                        required=True,
                        type=str,
                        help='Path to sample_files dir',
                        metavar='path/to/dataset')
    args = parser.parse_args()

    if not os.path.isdir(args.sample_files_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.sample_files_dir))
        exit()

    # Input depth
    scaled_depth_to_rgb_depth(args.sample_files_dir, INPUT_DEPTH_FILENAME, RGB_INPUT_DEPTH_FILENAME)

    # Output depth
    scaled_depth_to_rgb_depth(args.sample_files_dir, OUTPUT_DEPTH_FILENAME, RGB_OUTPUT_DEPTH_FILENAME)

    # Expected output depth
    scaled_depth_to_rgb_depth(args.sample_files_dir, EXPECTED_OUTPUT_DEPTH_FILENAME, RGB_EXPECTED_OUTPUT_DEPTH_FILENAME)

    # Normals
    normals_to_rgb_normals(args.sample_files_dir, NORMALS_FILENAME, RGB_NORMALS_FILENAME)

    # Occlusion Boundaries
    occlusion_to_rgb_occlusion(args.sample_files_dir, OCCLUSION_WEIGHTS_FILENAME, RGB_OCCLUSION_WEIGHTS_FILENAME)
