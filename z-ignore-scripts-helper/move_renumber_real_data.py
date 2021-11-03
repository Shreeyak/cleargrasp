import argparse
import fnmatch
import os
import shutil

EXT_COLOR_IMG = '-transparent-rgb-img.jpg'
EXT_DEPTH_IMG = '-transparent-depth-img.exr'
EXT_COLOR_OPAQUE_IMG = '-opaque-rgb-img.jpg'
EXT_DEPTH_GT = '-opaque-depth-img.exr'
EXT_MASK_ORIG = '-opaque-rgb-img.png'
EXT_MASK_NEW = '-mask.png'
NUM_UNIQUE_FILETYPES = 5


def get_indexes_in_dir(dataset_path, ext):
    '''Returns a list of numeric prefixes of all the rgb files present in dataset
    Eg, if our file is named 000000234-rgb.jpg, prefix is '000000234' and ext is '-rgb.jpg'


    Args:
        dataset_path (str): Path to dataset containing all the new files.
        ext (str): The extension after the prefix used to create list of indexes

    Returns:
        list: List of all indices, in str format
    '''
    dataset_prefixes = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in fnmatch.filter(files, '*' + ext):
            dataset_prefixes.append(filename[0:0 - len(ext)])
        break
    dataset_prefixes.sort()
    return dataset_prefixes


def main():
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
                        help='Path to destination dir. Files will be renamed and moved to here.',
                        metavar='path/to/result')
    parser.add_argument('-n',
                        '--num_start',
                        default=None,
                        type=int,
                        help='The initial value from which the numbering of renamed files must start. If not passed,' +
                        'will find last index in target dir and start numbering from that')
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.source_dir))
        exit()
    if not os.path.isdir(args.dest_dir):
        print('\nError: Source dir does not exist: {}\n'.format(args.dest_dir))
        exit()

    # Get list of indexes
    list_indexes_src = get_indexes_in_dir(args.source_dir, EXT_COLOR_IMG)
    list_indexes_tgt = get_indexes_in_dir(args.dest_dir, EXT_COLOR_IMG)

    # Move all files of that index from source to target
    if args.num_start is None:
        if list_indexes_tgt:
            new_initial_index = int(list_indexes_tgt[-1]) + 1
        else:
            new_initial_index = 0
    else:
        new_initial_index = args.num_start

    count_renamed = 0
    for i in range(len(list_indexes_src)):
        old_prefix_str = "{:09}".format(int(list_indexes_src[i]))
        new_prefix_str = "{:09}".format(new_initial_index + i)

        for root, dirs, files in os.walk(args.source_dir):
            list_files_with_prefix = fnmatch.filter(files, (old_prefix_str + '*'))
            if len(list_files_with_prefix) < NUM_UNIQUE_FILETYPES:
                print('Error: less than {} files with prefix {} in src dir'.format(NUM_UNIQUE_FILETYPES,
                                                                                   old_prefix_str))
                continue

            print("\tMoving files with prefix", old_prefix_str, "to", new_prefix_str)
            for filename in list_files_with_prefix:
                try:
                    # Rename the mask files while moving
                    if EXT_MASK_ORIG in filename:
                        new_filename = filename.replace(EXT_MASK_ORIG, EXT_MASK_NEW)
                        shutil.copy(os.path.join(args.source_dir, filename),
                                    os.path.join(args.dest_dir, new_filename.replace(old_prefix_str, new_prefix_str)))
                    else:
                        shutil.copy(os.path.join(args.source_dir, filename),
                                    os.path.join(args.dest_dir, filename.replace(old_prefix_str, new_prefix_str)))
                except shutil.Error as err:
                    print(err)
                count_renamed += 1
            break
    print('Finished: Moved {} files'.format(count_renamed))


if __name__ == "__main__":
    main()
