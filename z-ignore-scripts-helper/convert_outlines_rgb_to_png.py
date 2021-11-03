import argparse
import os
import glob

import imageio
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rearrange non-contiguous numbered images in a dataset, move to new folder and process.')

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
                        help='Path to dir of output depth',
                        metavar='path/to/dataset')

    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()
    if not os.path.isdir(args.dest_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()

    EXT_OUTLINE_RGB = '-outline-rgb.png'
    EXT_OUTLINE = '-outlineSegmentation.png'

    outline_files_list = sorted(glob.glob(os.path.join(args.source_dir, '*' + EXT_OUTLINE_RGB)))
    if len(outline_files_list) == 0:
        raise ValueError('No files in source dir')

    print('Going to convert {} files'.format(len(outline_files_list)))
    for outline_file in outline_files_list:
        outline_rgb = imageio.imread(outline_file)

        outline_png = np.zeros_like(outline_rgb)
        outline_png = outline_png[..., 0]

        outline_png[outline_rgb[..., 0] > 100] = 0
        outline_png[outline_rgb[..., 1] > 100] = 1
        outline_png[outline_rgb[..., 2] > 100] = 2

        prefix = os.path.basename(outline_file)[:9]
        outline_png_file = os.path.join(args.source_dir, prefix + EXT_OUTLINE)
        imageio.imwrite(outline_png_file, outline_png)

        print('Saved file: {}'.format(outline_png_file))