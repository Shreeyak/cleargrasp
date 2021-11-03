import os
import glob
import imageio
import numpy as np

# original results folder path
original_result_path_str = '/home/gani/deeplearning/brain/models/greppy-gbrain-transparent-detection/surface-normals-and-outlines-detection/pytorch-normals/data/exp-01-288p/results/test-real'
original_result_path = sorted(glob.glob(os.path.join(original_result_path_str, '*.jpg')))

# padded results folder path
padded_result_path_str = '/home/gani/deeplearning/brain/models/greppy-gbrain-transparent-detection/surface-normals-and-outlines-detection/pytorch-normals/data/exp-03-padded/results/test-real'
padded_result_path = sorted(glob.glob(os.path.join(padded_result_path_str, '*.jpg')))

# results folder path
result_folder_path_str = '/home/gani/deeplearning/brain/models/greppy-gbrain-transparent-detection/surface-normals-and-outlines-detection/pytorch-normals/data/results_collage'

for i in range(len(original_result_path)):
    original_file = imageio.imread(original_result_path[i])
    padded_file = imageio.imread(padded_result_path[i])

    # Create Vizualization of all the results
    grid_image = np.concatenate((original_file, padded_file), 0)
    result_viz_filename = ('%09d' + '-normals.jpg') % (i)
    imageio.imwrite(os.path.join(result_folder_path_str, result_viz_filename), grid_image)