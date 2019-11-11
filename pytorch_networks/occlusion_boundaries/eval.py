'''Run inference on outlines prediction model
'''

import os
import glob
import io
import argparse
import errno
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
from utils import utils

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run eval of outlines prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to yaml config file', metavar='path/to/config.yaml')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
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
    config_checkpoint_dict = CHECKPOINT['config']
    config_checkpoint = AttrDict(config_checkpoint_dict)
    print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))
else:
    raise ValueError('The checkpoint file does not have model_state_dict in it.\
                     Please use the newer checkpoint files!')

# Check for results store dir
# Create directory to save results
SUBDIR_RESULT = 'results'
SUBDIR_OUTLINES = 'outlines_files'

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
    os.makedirs(os.path.join(results_dir, SUBDIR_OUTLINES))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"\n'.format(results_dir), 'blue'))

###################### DataLoader #############################
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config_checkpoint.train.imgHeight,
        "width": config_checkpoint.train.imgWidth
    },
               interpolation='nearest')
])

# Make new dataloaders for each dataset
db_test_list = []
for dataset in config.eval.datasetsSynthetic:
    if dataset.images:
        label_dir = dataset.labels if dataset.labels else ''
        db = dataloader.OutlinesDataset(input_dir=dataset.images,
                                        label_dir=label_dir,
                                        transform=augs_test,
                                        input_only=None)
        db_test_list.append(db)

if db_test_list:
    db_test = torch.utils.data.ConcatDataset(db_test_list)
    testLoader = DataLoader(db_test,
                            batch_size=config.eval.batchSize,
                            shuffle=False,
                            num_workers=config.eval.numWorkers,
                            drop_last=False)
else:
    raise ValueError('No valid datasets were passed in config file')

###################### ModelBuilder #############################
if config_checkpoint.train.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='xception', sync_bn=True, freeze_bn=False)
elif config_checkpoint.train.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet', sync_bn=True, freeze_bn=False)
elif config_checkpoint.train.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='drn', sync_bn=True, freeze_bn=False)
else:
    raise ValueError('Invalid model passed')
model.load_state_dict(CHECKPOINT['model_state_dict'])

# Enable Multi-GPU training
print("Let's use", torch.cuda.device_count(), "GPUs!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model = model.to(device)
model.eval()

### Run Validation and Test Set ###
print('\nInference - Outline Prediction')
print('-' * 50 + '\n')
print('Running inference on Test sets at:\n    {}\n'.format(config.eval.datasetsSynthetic))

total_iou = 0.0
for ii, sample_batched in enumerate(tqdm(testLoader)):

    inputs, labels = sample_batched

    # Forward pass of the mini-batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    predictions = torch.max(outputs, 1)[1]
    _total_iou, per_class_iou, num_images_per_class = utils.get_iou(predictions,
                                                                    labels.long().squeeze(1),
                                                                    n_classes=config.train.numClasses)
    total_iou += _total_iou

    # Save output images, one at a time, to results
    img_tensor = inputs.detach().cpu()
    output_tensor = outputs.detach().cpu()
    label_tensor = labels.detach().cpu()

    # Extract each tensor within batch and save results
    for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor)):
        image, output, label = sample_batched

        result_path = os.path.join(results_dir, SUBDIR_RESULT,
                                    '{:09d}-outlines-result.png'.format(ii * config.eval.batchSize + iii))

        # Save Results
        # grid image with input, prediction and label
        output_prediction = torch.unsqueeze(torch.max(output, 0)[1].float(), 0)
        output_prediction_rgb = utils.label_to_rgb(output_prediction)
        label_rgb = utils.label_to_rgb(label)

        grid_image = torch.cat((image, output_prediction_rgb, label_rgb), dim=2)
        grid_image = make_grid(grid_image, 1, normalize=True, scale_each=True)
        numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
        numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)
        imageio.imwrite(result_path, numpy_grid)

        # Save the Occlusion Weights file used by depth2depth
        # calculating occlusion weights
        output_softmax = nn.Softmax(dim=1)(output).numpy()
        weight = (1 - output_softmax[1, :, :])
        x = np.power(weight, 3)
        x = np.multiply(x, 1000)
        final_weight = x.astype(np.uint16)
        # Increase the min and max values by small amount epsilon so that absolute min/max values
        # don't cause problems in the depth2depth optimization code.
        eps = 1
        final_weight[final_weight == 0] += eps
        final_weight[final_weight == 1000] -= eps
        # Save the weights file
        array_buffer = final_weight.tobytes()
        img = Image.new("I", final_weight.T.shape)
        img.frombytes(array_buffer, 'raw', 'I;16')
        result_weights_path = os.path.join(results_dir, SUBDIR_OUTLINES,
                                           '{:09d}-occlusion-weight.png'.format(ii * config.eval.batchSize + iii))
        img.save(result_weights_path)

        # Save the weights' visualization
        final_weight_color = (weight * 255).astype(np.uint8)
        final_weight_color = np.expand_dims(final_weight_color, axis=2)
        final_weight_color = cv2.applyColorMap(final_weight_color, cv2.COLORMAP_OCEAN)
        final_weight_color = cv2.cvtColor(final_weight_color, cv2.COLOR_BGR2RGB)
        result_weights_viz_path = os.path.join(
            results_dir, SUBDIR_OUTLINES, '{:09d}-occlusion-weight-rgb.png'.format(ii * config.eval.batchSize + iii))
        imageio.imwrite(result_weights_viz_path, final_weight_color)

        # Save overlay of result on RGB image.
        input_image = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask_rgb = input_image.copy()

        pred = output_prediction.squeeze(0).numpy()
        mask_rgb[pred == 1, 1] = 255
        mask_rgb[pred == 2, 0] = 255
        masked_img = cv2.addWeighted(mask_rgb, 0.6, input_image, 0.4, 0)
        result_path = os.path.join(
            results_dir, SUBDIR_OUTLINES, '{:09d}-overlaid-rgb.png'.format(ii * config.eval.batchSize + iii))
        imageio.imwrite(result_path, masked_img)

        # Save RGB viz of Result
        rgb_viz = np.zeros_like(input_image)
        rgb_viz[pred == 0, 0] = 255
        rgb_viz[pred == 1, 1] = 255
        rgb_viz[pred == 2, 2] = 255
        rgb_viz = cv2.resize(rgb_viz, (512, 288), interpolation=cv2.INTER_NEAREST)
        result_path = os.path.join(
            results_dir, SUBDIR_OUTLINES, '{:09d}-outline-rgb.png'.format(ii * config.eval.batchSize + iii))
        imageio.imwrite(result_path, rgb_viz)

    miou = total_iou / ((len(testLoader)) * config.eval.batchSize)
    print('Test Mean IoU: {:.4f}'.format(miou))
