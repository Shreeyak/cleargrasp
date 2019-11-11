'''Train unet for surface normals
'''

import argparse
import csv
import errno
import glob
import io
import os
import shutil

from attrdict import AttrDict
import cv2
import h5py
import imageio
import imgaug as ia
import numpy as np
import torch
import torch.nn as nn
import yaml
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from termcolor import colored
from tqdm import tqdm

import dataloader
from modeling import deeplab
from utils import utils

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run eval of outlines prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to yaml config file', metavar='path/to/config.yaml')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.safe_load(fd)
config = AttrDict(config_yaml)

print('Inference of Masks model. Loading checkpoint...')

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
SUBDIR_MASKS = 'masks'

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
    os.makedirs(os.path.join(results_dir, SUBDIR_MASKS))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"\n'.format(results_dir), 'blue'))

# Create CSV File to store error metrics
csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)
field_names = ["Image Num", "mIoU", "TP", "TN", "FP", "FN"]
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
        if dataset.images:
            print('dataset.images', dataset.images)
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_synthetic.append(db)

# Make new dataloaders for each real dataset
db_test_list_real = []
if config.eval.datasetsReal is not None:
    for dataset in config.eval.datasetsReal:
        if dataset.images:
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.masks,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list_real.append(db)

if len(db_test_list_synthetic) + len(db_test_list_real) == 0:
    raise ValueError('No valid datasets provided to run inference on!')

if db_test_list_synthetic:
    db_test_synthetic = torch.utils.data.ConcatDataset(db_test_list_synthetic)
    testLoader_synthetic = DataLoader(db_test_synthetic,
                                      batch_size=config.eval.batchSize,
                                      shuffle=False,
                                      num_workers=config.eval.numWorkers,
                                      drop_last=False)

if db_test_list_real:
    db_test_real = torch.utils.data.ConcatDataset(db_test_list_real)
    testLoader_real = DataLoader(db_test_real,
                                 batch_size=config.eval.batchSize,
                                 shuffle=False,
                                 num_workers=config.eval.numWorkers,
                                 drop_last=False)

###################### ModelBuilder #############################
if config.eval.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='xception', sync_bn=True, freeze_bn=False)
elif config.eval.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='resnet', sync_bn=True, freeze_bn=False)
elif config.eval.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone='drn', sync_bn=True, freeze_bn=False)
else:
    raise ValueError('Invalid model "{}" in config file. Must be one of ["deeplab_xception", "deeplab_resnet"]'.format(
        config.eval.model))

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
criterion = nn.CrossEntropyLoss(size_average=False, reduce=True)

### Run Validation and Test Set ###
print('\nInference - Masks of Transparent Objects')
print('-' * 50 + '\n')
print('Running inference on Test sets at:\n    {}\n    {}\n'.format(config.eval.datasetsReal,
                                                                    config.eval.datasetsSynthetic))
print('Results will be saved to:\n    {}\n'.format(config.eval.resultsDir))

dataloaders_dict = {}
if db_test_list_real:
    dataloaders_dict.update({'real': testLoader_real})
if db_test_list_synthetic:
    dataloaders_dict.update({'synthetic': testLoader_synthetic})

for key in dataloaders_dict:
    print('\n' + key + ':')
    print('=' * 30)

    testLoader = dataloaders_dict[key]

    if len(testLoader.dataset) == 0:
        continue

    running_loss = 0.0
    total_iou = 0.0

    running_iou = []
    running_tp = []
    running_tn = []
    running_fp = []
    running_fn = []
    for ii, sample_batched in enumerate(tqdm(testLoader)):

        inputs, labels = sample_batched

        # Forward pass of the mini-batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        predictions = torch.max(outputs, 1)[1]
        loss = criterion(outputs, labels.long().squeeze(1))

        running_loss += loss.item()

        _total_iou, per_class_iou, num_images_per_class = utils.get_iou(predictions,
                                                                        labels.long().squeeze(1),
                                                                        n_classes=config.eval.numClasses)
        total_iou += _total_iou

        # print('Batch {:09d} Loss: {:.4f}'.format(ii, loss.item()))

        # Save output images, one at a time, to results
        img_tensor = inputs.detach().cpu()
        output_tensor = outputs.detach().cpu()
        label_tensor = labels.detach().cpu()

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor)):
            img, output, label = sample_batched

            pred = torch.max(output, 0)[1].float()

            # print('pred:', pred.shape, pred.dtype, pred.min(), pred.max())
            # print('label:', label.shape, label.dtype, label.min(), label.max())
            iou, tp, tn, fp, fn = utils.compute_metrics(pred, label.squeeze(0))
            running_iou.append(iou)
            running_tp.append(tp)
            running_tn.append(tn)
            running_fp.append(fp)
            running_fn.append(fn)

            # Write the data into a csv file
            with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [((ii * config.eval.batchSize) + iii), iou, tp, tn, fp, fn]
                writer.writerow(dict(zip(field_names, row_data)))

            result_path = os.path.join(results_dir, SUBDIR_RESULT,
                                       '{:09d}-mask-result.png'.format(ii * config.eval.batchSize + iii))

            # Save Results
            # grid image with input, prediction and label
            pred_rgb = utils.label_to_rgb(torch.unsqueeze(pred, 0))
            label_rgb = utils.label_to_rgb(label)

            images = torch.cat((img, pred_rgb, label_rgb), dim=2)
            grid_image = make_grid(images, 1, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)
            imageio.imwrite(result_path, numpy_grid)

            result_mask_path = os.path.join(results_dir, SUBDIR_MASKS,
                                            '{:09d}-mask.png'.format(ii * config.eval.batchSize + iii))
            imageio.imwrite(result_mask_path, (pred.squeeze(0).numpy() * 255).astype(np.uint8))

    epoch_loss = running_loss / (len(testLoader))
    print('\nTest Mean Loss: {:.4f}'.format(epoch_loss))
    miou = total_iou / ((len(testLoader)) * config.eval.batchSize)
    print('Test Mean IoU: {:.4f}'.format(miou))

    num_images = len(testLoader.dataset)  # Num of total images
    miou = round(sum(running_iou) / num_images, 2)
    mtp = round(sum(running_tp) / num_images, 2)
    mtn = round(sum(running_tn) / num_images, 2)
    mfp = round(sum(running_fp) / num_images, 2)
    mfn = round(sum(running_fn) / num_images, 2)
    print(
        '\nTest Metrics - mIoU: {:.2f}, TP: {:.2f}%, TN: {:.2f}%, FP: {:.2f}%, FN: {:.2f}%, num_images: {}\n\n'.format(
            miou, mtp, mtn, mfp, mfn, num_images))
