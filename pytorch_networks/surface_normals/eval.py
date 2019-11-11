'''Train unet for surface normals
'''
import argparse
import csv
import errno
import os
import glob
import io
import shutil

from termcolor import colored
import yaml
from attrdict import AttrDict
import imageio
import numpy as np
import h5py
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

from modeling import deeplab
import dataloader
import loss_functions
from utils import utils
import dataloader_matterport

print('Inference of Surface Normal Estimation model. Loading checkpoint...')

parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

###################### Load Config File #############################
CONFIG_FILE_PATH = args.configFile  #'config/config.yaml'
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

###################### Load Checkpoint and its data #############################
if not os.path.isfile(config.eval.pathWeightsFile):
    raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
        config.eval.pathWeightsFile))

# Read config file stored in the model checkpoint to re-use it's params
CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
if 'model_state_dict' in CHECKPOINT:
    print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))

    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

# Create directory to save results
SUBDIR_RESULT = 'results'
SUBDIR_NORMALS = 'normals_files'

results_root_dir = config.eval.resultsDir
runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
if os.path.isdir(os.path.join(results_dir, SUBDIR_RESULT)):
    NUM_FILES_IN_EMPTY_FOLDER = 0
    if len(os.listdir(os.path.join(results_dir, SUBDIR_RESULT))) > NUM_FILES_IN_EMPTY_FOLDER:
        prev_run_id += 1
        results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
        os.makedirs(results_dir)
else:
    os.makedirs(results_dir)

try:
    os.makedirs(os.path.join(results_dir, SUBDIR_RESULT))
    os.makedirs(os.path.join(results_dir, SUBDIR_NORMALS))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"\n'.format(results_dir), 'blue'))

# Create CSV File to store error metrics
csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)
field_names = ["Image Num", "Mean", "Median", "<11.25", "<22.5", "<30"]
with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
    writer.writeheader()

###################### DataLoader #############################
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config.eval.imgHeight,
        "width": config.eval.imgWidth
    }, interpolation='nearest'),
])

# Make new dataloaders for each synthetic dataset
db_test_list_synthetic = []
if config.eval.datasetsSynthetic is not None:
    for dataset in config.eval.datasetsSynthetic:
        print('Creating Synthetic Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_synthetic.append(db)

# Make new dataloaders for each real dataset
db_test_list_real = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        print('Creating Real Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_real.append(db)

db_test_list_matterport = []
if config.eval.datasetsMatterport is not None:
    for dataset in config.eval.datasetsMatterport:
        print('Creating Matterport Images dataset from: "{}"'.format(dataset.images))
        if dataset.images:
            db = dataloader_matterport.SurfaceNormalsDataset(input_dir=dataset.images,
                                                             label_dir=dataset.labels,
                                                             transform=augs_test,
                                                             input_only=None)
            db_test_list_matterport.append(db)

# Create pytorch dataloaders from datasets
dataloaders_dict = {}
if db_test_list_synthetic:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
    testLoader_synthetic = DataLoader(db_test_synthetic,
                                      batch_size=config.eval.batchSize,
                                      shuffle=False,
                                      num_workers=config.eval.numWorkers,
                                      drop_last=False)
    dataloaders_dict.update({'synthetic': testLoader_synthetic})

if db_test_list_real:
    db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)
    testLoader_real = DataLoader(db_test_real,
                                 batch_size=config.eval.batchSize,
                                 shuffle=False,
                                 num_workers=config.eval.numWorkers,
                                 drop_last=False)
    dataloaders_dict.update({'real': testLoader_real})

if db_test_list_matterport:
    db_test_matterport = torch.utils.data.ConcatDataset(db_test_list_matterport)
    testLoader_matterport = DataLoader(db_test_matterport,
                                       batch_size=config.eval.batchSize,
                                       shuffle=False,
                                       num_workers=config.eval.numWorkers,
                                       drop_last=False)
    dataloaders_dict.update({'matterport': testLoader_matterport})

assert (len(dataloaders_dict) > 0), 'No valid datasets given in config.yaml to run inference on!'

###################### ModelBuilder #############################
if config.eval.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='xception', sync_bn=True,
                            freeze_bn=False)
elif config.eval.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='resnet', sync_bn=True,
                            freeze_bn=False)
elif config.eval.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='drn', sync_bn=True,
                            freeze_bn=False)

model.load_state_dict(CHECKPOINT['model_state_dict'])

# Enable Multi-GPU training
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

### Select Loss Func ###
if config_checkpoint.train.lossFunc == 'cosine':
    criterion = loss_functions.loss_fn_cosine
elif config_checkpoint.train.lossFunc == 'radians':
    criterion = loss_functions.loss_fn_radians
else:
    raise ValueError('Invalid lossFunc from config file. Can only be "cosine" or "radians".\
                     Value passed is: {}'.format(config_checkpoint.train.lossFunc))

### Run Validation and Test Set ###
print('\nInference - Surface Normal Estimation')
print('-' * 50 + '\n')
print(colored('Results will be saved to: {}\n'.format(config.eval.resultsDir), 'green'))

for key in dataloaders_dict:
    print('Running inference on {} dataset:'.format(key))
    print('=' * 30)

    running_loss = 0.0
    running_mean = []
    running_median = []
    running_percentage1 = []
    running_percentage2 = []
    running_percentage3 = []

    testLoader = dataloaders_dict[key]
    for ii, sample_batched in enumerate(tqdm(testLoader)):
        # NOTE: In raw data, invalid surface normals are represented by [-1, -1, -1]. However, this causes
        #       problems during normalization of vectors. So they are represented as [0, 0, 0] in our dataloader output.

        inputs, labels, masks = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

        loss = criterion(normal_vectors_norm, labels, reduction='elementwise_mean')
        running_loss += loss.item()

        # loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
        #     normal_vectors_norm, labels)
        # print('Batch {:09d} Mean: {:.4f} deg'.format(ii, loss_deg_mean.item()))
        # print('Batch {:09d} Median: {:.4f} deg'.format(ii, loss_deg_median.item()))
        # print('Batch {:09d} P1: {:.4f} %'.format(ii, percentage_1.item()))
        # print('Batch {:09d} P2: {:.4f} %'.format(ii, percentage_2.item()))
        # print('Batch {:09d} P3: {:.4f} %'.format(ii, percentage_3.item()))
        # print('Batch {:09d} num_images: {:d}'.format(ii, labels.shape[0]))

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = normal_vectors_norm.detach().cpu()
        label_tensor = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor, mask_tensor)):
            img, output, label, mask = sample_batched

            # Calc metrics
            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = loss_functions.metric_calculator(
                output, label, mask=mask)
            running_mean.append(loss_deg_mean.item())
            running_median.append(loss_deg_median.item())
            running_percentage1.append(percentage_1.item())
            running_percentage2.append(percentage_2.item())
            running_percentage3.append(percentage_3.item())

            # Write the data into a csv file
            with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [((ii * config.eval.batchSize) + iii),
                            loss_deg_mean.item(),
                            loss_deg_median.item(),
                            percentage_1.item(),
                            percentage_2.item(),
                            percentage_3.item()]
                writer.writerow(dict(zip(field_names, row_data)))

            # Save grid image with input, prediction and label
            masks_3d = torch.stack((mask, mask, mask), dim=0)
            grid_image = make_grid([img, output, label, masks_3d], 4, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

            result_path = os.path.join(results_dir, SUBDIR_RESULT,
                                       '{:09d}-normals-result.jpg'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(result_path, numpy_grid)

            # # Write Predicted Surface Normal as hdf5 file for depth2depth
            # # NOTE: The hdf5 expected shape is (3, height, width), dtype float32
            # output_path_hdf5 = os.path.join(results_dir, SUBDIR_NORMALS,
            #                                 '{:09d}-normals.h5'.format(ii * config.eval.batchSize + iii))
            # with h5py.File(output_path_hdf5, "w") as f:
            #     dset = f.create_dataset('/result', data=output.numpy())

            # Save PNG and EXR Output
            output_rgb = utils.normal_to_rgb(output.numpy().transpose((1, 2, 0)))
            output_rgb *= 255
            output_rgb = output_rgb.astype(np.uint8)
            output_rgb = cv2.resize(output_rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
            output_path_rgb = os.path.join(results_dir, SUBDIR_NORMALS,
                                           '{:09d}-normals.png'.format(ii * config.eval.batchSize + iii))
            output_path_exr = os.path.join(results_dir, SUBDIR_NORMALS,
                                           '{:09d}-normals.exr'.format(ii * config.eval.batchSize + iii))
            output_path_valid_mask = os.path.join(results_dir, SUBDIR_NORMALS,
                                                  '{:09d}-valid-mask.png'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(output_path_rgb, output_rgb)
            utils.exr_saver(output_path_exr, output.numpy())

            mask_valid_pixels = (mask_valid_pixels.numpy() * 255).astype(np.uint8)
            imageio.imwrite(output_path_valid_mask, mask_valid_pixels)

    num_batches = len(testLoader)  # Num of batches
    num_images = len(testLoader.dataset)  # Num of total images
    print('\nnum_batches:', num_batches)
    print('num_images:', num_images)
    epoch_loss = running_loss / num_batches
    print('Test Mean Loss: {:.4f}'.format(epoch_loss))

    epoch_mean = sum(running_mean) / num_images
    epoch_median = sum(running_median) / num_images
    epoch_percentage1 = sum(running_percentage1) / num_images
    epoch_percentage2 = sum(running_percentage2) / num_images
    epoch_percentage3 = sum(running_percentage3) / num_images
    print(
        '\nTest Metrics - Mean: {:.2f}deg, Median: {:.2f}deg, P1: {:.2f}%, P2: {:.2f}%, p3: {:.2f}%, num_images: {}\n\n'
        .format(epoch_mean, epoch_median, epoch_percentage1, epoch_percentage2, epoch_percentage3, num_images))
