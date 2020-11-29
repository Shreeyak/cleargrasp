'''Train unet for surface normals
'''

import argparse
import errno
import glob
import io
import os
import random
import shutil

import cv2
import imgaug as ia
import numpy as np
import oyaml
import torch
import torch.nn as nn
from attrdict import AttrDict
from imgaug import augmenters as iaa
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dataloader
import dataloader_matterport
import dataloader_scannet
import loss_functions
from modeling import deeplab
from utils import utils

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run training of outlines prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)
# print(colored('Config being used for training:\n{}\n\n'.format(oyaml.dump(config_yaml)), 'green'))

###################### Logs (TensorBoard)  #############################
# Create directory to save results
SUBDIR_RESULT = 'checkpoints'

results_root_dir = config.train.logsDir
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
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

MODEL_LOG_DIR = results_dir
CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, SUBDIR_RESULT)
shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
print('Saving results to folder: ' + colored('"{}"'.format(results_dir), 'blue'))

# Create a tensorboard object and Write config to tensorboard
writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

string_out = io.StringIO()
oyaml.dump(config_yaml, string_out, default_flow_style=False)
config_str = string_out.getvalue().split('\n')
string = ''
for line in config_str:
    string = string + '    ' + line + '\n\r'
writer.add_text('Config', string, global_step=None)

###################### DataLoader #############################
# Train Dataset - Create a dataset object for each dataset in our list, Concatenate datasets, select subset for training
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),

    # Bright Patches
    iaa.Sometimes(
        0.1,
        iaa.blend.Alpha(factor=(0.2, 0.7),
                        first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                          upscale_method='cubic',
                                                          iterations=(1, 2)),
                        name="simplex-blend")),

    # Color Space Mods
    iaa.Sometimes(
        0.3,
        iaa.OneOf([
            iaa.Add((20, 20), per_channel=0.7, name="add"),
            iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                               name="hue"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                               name="sat"),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
            iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        ])),

    # Blur and Noise
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, None), [
            iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                       iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
            iaa.OneOf([
                iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
            ]),
        ],
                   random_order=True)),

    # Colored Blocks
    iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
])
input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]

db_synthetic_lst = []
if config.train.datasetsTrain is not None:
    for dataset in config.train.datasetsTrain:
        if dataset.images:
            db_synthetic = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                            label_dir=dataset.labels,
                                                            transform=augs_train,
                                                            input_only=input_only)
            db_synthetic_lst.append(db_synthetic)
    db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)

#Note: If multiple folders for matterport/scannet are passed, only the last one will be taken in.
if config.train.datasetsMatterportTrain is not None:
    for dataset in config.train.datasetsMatterportTrain:
        if dataset.images:
            db_matterport = dataloader_matterport.SurfaceNormalsDataset(input_dir=dataset.images,
                                                                        label_dir=dataset.labels,
                                                                        transform=augs_train,
                                                                        input_only=input_only)
            print('Num of Matterport imgs:', len(db_matterport))

if config.train.datasetsScannetTrain is not None:
    for dataset in config.train.datasetsScannetTrain:
        if dataset.images:
            db_scannet = dataloader_scannet.SurfaceNormalsDataset(input_dir=dataset.images,
                                                                  label_dir=dataset.labels,
                                                                  transform=augs_train,
                                                                  input_only=input_only)
            print('Num of ScanNet imgs:', len(db_scannet))

# Validation Dataset
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
])

db_val_list = []
if config.train.datasetsVal is not None:
    for dataset in config.train.datasetsVal:
        if dataset.images:
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  transform=augs_test,
                                                  input_only=None)
            train_size = int(config.train.percentageDataForValidation * len(db))
            db = torch.utils.data.Subset(db, range(train_size))
            db_val_list.append(db)

if config.train.datasetsMatterportVal is not None:
    for dataset in config.train.datasetsMatterportVal:
        if dataset.images:
            db = dataloader_matterport.SurfaceNormalsDataset(input_dir=dataset.images,
                                                             label_dir=dataset.labels,
                                                             transform=augs_test,
                                                             input_only=None)
            train_size = int(config.train.percentageDataForMatterportVal * len(db))
            db = torch.utils.data.Subset(db, range(train_size))
            db_val_list.append(db)

if config.train.datasetsScannetVal is not None:
    for dataset in config.train.datasetsScannetVal:
        if dataset.images:
            db = dataloader_scannet.SurfaceNormalsDataset(input_dir=dataset.images,
                                                          label_dir=dataset.labels,
                                                          transform=augs_test,
                                                          input_only=None)
            train_size = int(config.train.percentageDataForScannettVal * len(db))
            db = torch.utils.data.Subset(db, range(train_size))
            db_val_list.append(db)

if db_val_list:
    db_val = torch.utils.data.ConcatDataset(db_val_list)

# Test Dataset - Real
db_test_list = []
if config.train.datasetsTestReal is not None:
    for dataset in config.train.datasetsTestReal:
        if dataset.images:
            mask_dir = dataset.masks if hasattr(dataset, 'masks') and dataset.masks else ''
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  mask_dir=mask_dir,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_list.append(db)
    if db_test_list:
        db_test = torch.utils.data.ConcatDataset(db_test_list)

# Test Dataset - Synthetic
db_test_synthetic_list = []
if config.train.datasetsTestSynthetic is not None:
    for dataset in config.train.datasetsTestSynthetic:
        if dataset.images:
            db = dataloader.SurfaceNormalsDataset(input_dir=dataset.images,
                                                  label_dir=dataset.labels,
                                                  transform=augs_test,
                                                  input_only=None)
            db_test_synthetic_list.append(db)
    if db_test_synthetic_list:
        db_test_synthetic = torch.utils.data.ConcatDataset(db_test_synthetic_list)

# Create dataloaders
if db_val_list:
    assert (config.train.validationBatchSize <= len(db_val)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(config.train.validationBatchSize, len(db_val))

    validationLoader = DataLoader(db_val,
                                  batch_size=config.train.validationBatchSize,
                                  shuffle=True,
                                  num_workers=config.train.numWorkers,
                                  drop_last=False)
if db_test_list:
    assert (config.train.testBatchSize <= len(db_test)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test))

    testLoader = DataLoader(db_test,
                            batch_size=config.train.testBatchSize,
                            shuffle=False,
                            num_workers=config.train.numWorkers,
                            drop_last=True)
if db_test_synthetic_list:
    assert (config.train.testBatchSize <= len(db_test_synthetic)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(config.train.testBatchSize, len(db_test_synthetic_list))

    testSyntheticLoader = DataLoader(db_test_synthetic,
                                     batch_size=config.train.testBatchSize,
                                     shuffle=True,
                                     num_workers=config.train.numWorkers,
                                     drop_last=True)


# Resize Tensor
def resize_tensor(input_tensor, height, width):
    augs_label_resize = iaa.Sequential([iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_label_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array


###################### ModelBuilder #############################
if config.train.model == 'deeplab_xception':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='xception', sync_bn=True,
                            freeze_bn=False)
elif config.train.model == 'deeplab_resnet':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet', sync_bn=True,
                            freeze_bn=False)
elif config.train.model == 'drn':
    model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='drn', sync_bn=True,
                            freeze_bn=False)  # output stride is 8 for drn
else:
    raise ValueError(
        'Invalid model "{}" in config file. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'
        .format(config.train.model))

if config.train.continueTraining:
    print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(config.train.pathPrevCheckpoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(config.train.pathPrevCheckpoint))

    CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

    if 'model_state_dict' in CHECKPOINT:
        # Our weights file with various dicts
        model.load_state_dict(CHECKPOINT['model_state_dict'])
    elif 'state_dict' in CHECKPOINT:
        # Original Author's checkpoint
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        # Our old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)

# Enable Multi-GPU training
print("Let's use", torch.cuda.device_count(), "GPUs!")
model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

###################### Setup Optimizer #############################
if config.train.model == 'unet':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.train.optimAdam.learningRate),
                                 weight_decay=float(config.train.optimAdam.weightDecay))
elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
elif config.train.model == 'refinenet' or config.train.model == 'densenet':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config.train.optimSgd.learningRate),
                                momentum=float(config.train.optimSgd.momentum),
                                weight_decay=float(config.train.optimSgd.weight_decay))
else:
    raise ValueError(
        'Invalid model "{}" in config file for optimizer. Must be one of ["drn", unet", "deeplab_xception", "deeplab_resnet", "refinenet"]'
        .format(config.train.model))

if not config.train.lrScheduler:
    pass
elif config.train.lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.train.lrSchedulerStep.step_size,
                                                   gamma=float(config.train.lrSchedulerStep.gamma))
elif config.train.lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=float(config.train.lrSchedulerPlateau.factor),
                                                              patience=config.train.lrSchedulerPlateau.patience,
                                                              verbose=True)
elif config.train.lrScheduler == 'lr_poly':
    pass
else:
    raise ValueError(
        "Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(
            config.train.lrScheduler))

# Continue Training from prev checkpoint if required
if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
    if 'optimizer_state_dict' in CHECKPOINT:
        optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
    else:
        print(
            colored(
                'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))

### Select Loss Func ###
if config.train.lossFunc == 'cosine':
    criterion = loss_functions.loss_fn_cosine
elif config.train.lossFunc == 'radians':
    criterion = loss_functions.loss_fn_radians
else:
    raise ValueError("Invalid lossFunc from config file. Can only be ['cosine', 'radians']. " +
                     "Value passed is: {}".format(config.train.lossFunc))

###################### Train Model #############################
# Set total iter_num (number of batches seen by model, used for logging)
total_iter_num = 0
START_EPOCH = 0
END_EPOCH = config.train.numEpochs

if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
    if 'model_state_dict' in CHECKPOINT:
        # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
        total_iter_num = CHECKPOINT['total_iter_num'] + 1
        START_EPOCH = CHECKPOINT['epoch'] + 1
        END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
    else:
        print(
            colored(
                'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                       Starting from epoch num 0', 'red'))

for epoch in range(START_EPOCH, END_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
    print('-' * 30)

    # Log the current Epoch Number
    writer.add_scalar('data/Epoch Number', epoch, total_iter_num)

    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)

    # Update Learning Rate Scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler.step()
    elif config.train.lrScheduler == 'ReduceLROnPlateau':
        lr_scheduler.step(epoch_loss)
    elif config.train.lrScheduler == 'lr_poly':
        if epoch % config.train.epochSize == config.train.epochSize - 1:
            lr_ = utils.lr_poly(float(config.train.optimSgd.learningRate), int(epoch - START_EPOCH),
                                int(END_EPOCH - START_EPOCH), 0.9)

            train_params = model.parameters()
            if model == 'drn':
                train_params = [{
                    'params': model.get_1x_lr_params(),
                    'lr': config.train.optimSgd.learningRate
                }, {
                    'params': model.get_10x_lr_params(),
                    'lr': config.train.optimSgd.learningRate * 10
                }]

            optimizer = torch.optim.SGD(train_params,
                                        lr=lr_,
                                        momentum=float(config.train.optimSgd.momentum),
                                        weight_decay=float(config.train.optimSgd.weight_decay))

    # split scannet and matterport dataset for  random selection of data
    db_train_list = []
    if config.train.datasetsTrain is not None:
        if config.train.datasetsTrain[0].images:
            train_size_synthetic = int(config.train.percentageDataForTraining * len(db_synthetic))
            db, _ = torch.utils.data.random_split(db_synthetic,
                                                  (train_size_synthetic, len(db_synthetic) - train_size_synthetic))
            db_train_list.append(db)

    if config.train.batchSizeMatterport == 0 and config.train.batchSizeScannet == 0:
        if config.train.datasetsMatterportTrain is not None:
            if config.train.datasetsMatterportTrain[0].images:
                train_size_matterport = int(config.train.percentageDataForMatterportTraining * len(db_matterport))
                db, _ = torch.utils.data.random_split(
                    db_matterport, (train_size_matterport, len(db_matterport) - train_size_matterport))
                db_train_list.append(db)

        if config.train.datasetsScannetTrain is not None:
            if config.train.datasetsScannetTrain[0].images:
                train_size_scannet = int(config.train.percentageDataForScannetTraining * len(db_scannet))
                db, _ = torch.utils.data.random_split(db_scannet,
                                                      (train_size_scannet, len(db_scannet) - train_size_scannet))
                db_train_list.append(db)
    else:
        # Create separate loaders for matterport and scannet. Will call them and add to batch manually.
        if config.train.datasetsMatterportTrain is not None:
            trainLoaderMatterport = DataLoader(db_matterport,
                                               batch_size=config.train.batchSizeMatterport,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True,
                                               pin_memory=True)
            loaderMatterport = iter(trainLoaderMatterport)

        if config.train.datasetsScannetTrain is not None:
            trainLoaderScannet = DataLoader(db_scannet,
                                            batch_size=config.train.batchSizeScannet,
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,
                                            pin_memory=True)
            loaderScannet = iter(trainLoaderScannet)

    db_train = torch.utils.data.ConcatDataset(db_train_list)
    trainLoader = DataLoader(db_train,
                             batch_size=config.train.batchSize,
                             shuffle=True,
                             num_workers=config.train.numWorkers,
                             drop_last=True,
                             pin_memory=True)

    model.train()

    running_loss = 0.0
    running_mean = 0
    running_median = 0
    for iter_num, batch in enumerate(tqdm(trainLoader)):
        total_iter_num += 1

        # Get data
        if config.train.batchSizeMatterport == 0 and config.train.batchSizeScannet == 0:
            inputs, labels, masks = batch
        else:
            inputs_t, labels_t, masks_t = batch
            inputs_m, labels_m = next(loaderMatterport)
            inputs_s, labels_s = next(loaderScannet)
            inputs = torch.cat([inputs_t, inputs_m, inputs_s], dim=0)
            labels = torch.cat([labels_t, labels_m, labels_s], dim=0)
            masks = torch.cat([masks_t, torch.ones_like(masks_t), torch.ones_like(masks_t)], dim=0)

        if config.train.model == 'refinenet':
            labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
            labels_resized = labels_resized.to(device)
        if config.train.model == 'densenet':
            labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
            labels_resized = labels_resized.to(device)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get Model Graph
        # if epoch == 0 and iter_num == 0:
        # writer.add_graph(model, inputs, False)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        # normal_vectors_norm = normal_vectors_norm.double()

        if config.train.model == 'unet':
            loss = criterion(normal_vectors_norm, labels, reduction='sum')
        elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
            loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
            loss /= config.train.batchSize
        elif config.train.model == 'refinenet' or config.train.model == 'densenet':
            # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
            loss = criterion(normal_vectors_norm, labels_resized, reduction='sum')
            loss /= config.train.batchSize
        loss.backward()
        optimizer.step()

        normal_vectors_norm = normal_vectors_norm.detach().cpu()
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels.double(), mask_tensor)
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()

        # statistics
        running_loss += loss.item()
        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
        writer.add_scalar('data/Train Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
        writer.add_scalar('data/Train Median Error (deg)', loss_deg_median.item(), total_iter_num)

        # Log 10 images every N iters
        if (iter_num % config.train.saveImageIntervalIter) == 0:
            if config.train.model == 'refinenet':
                output = resize_tensor(normal_vectors_norm, int(normal_vectors_norm.shape[2] * 4),
                                       int(normal_vectors_norm.shape[3] * 4))
            elif config.train.model == 'densenet':
                output = resize_tensor(normal_vectors_norm, int(normal_vectors_norm.shape[2] * 2),
                                       int(normal_vectors_norm.shape[3] * 2))
            else:
                output = normal_vectors_norm

            grid_image = utils.create_grid_image(inputs, output.float(), labels, max_num_images_to_save=16)
            writer.add_image('Train', grid_image, total_iter_num)

    # Log Epoch Loss
    num_samples = (len(trainLoader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
    epoch_mean = running_mean / num_samples
    epoch_median = running_median / num_samples
    print('Train Epoch Mean Error (deg): {:.4f}'.format(epoch_mean))
    print('Train Epoch Median Error (deg): {:.4f}'.format(epoch_median))

    # Log Current Learning Rate
    # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
    #    The value read here will only reflect the initial lr value.
    current_learning_rate = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_learning_rate, total_iter_num)

    # Log 10 images every N epochs
    if (epoch % config.train.saveImageInterval) == 0:
        if config.train.model == 'refinenet':
            output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                   int(normal_vectors_norm.shape[3] * 4))
        elif config.train.model == 'densenet':
            output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                   int(normal_vectors_norm.shape[3] * 2))
        else:
            output = normal_vectors_norm.detach().cpu()

        grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                             output.float(),
                                             labels.detach().cpu(),
                                             max_num_images_to_save=16)
        writer.add_image('Train', grid_image, total_iter_num)

    # Save the model checkpoint every N epochs
    if (epoch % config.train.saveModelInterval) == 0:
        filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
        if torch.cuda.device_count() > 1:
            model_params = model.module.state_dict()  # Saving nn.DataParallel model
        else:
            model_params = model.state_dict()

        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': total_iter_num,
                'epoch_loss': epoch_loss,
                'config': config_yaml
            }, filename)

    ###################### Validation Cycle #############################
    if db_val_list:
        print('\nValidation:')
        print('=' * 10)

        model.eval()

        running_loss = 0.0
        running_mean = 0
        running_median = 0
        for iter_num, sample_batched in enumerate(tqdm(validationLoader)):
            inputs, labels, masks = sample_batched

            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

            if config.train.model == 'unet':
                loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
            elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                loss /= config.train.batchSize
            elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                normal_vectors_norm = normal_vectors_norm.detach().cpu()
                labels_resized = labels_resized.detach().cpu()
                # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                loss = criterion(normal_vectors_norm, labels_resized, reduction='sum')
                loss /= config.train.batchSize

            running_loss += loss.item()

            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
                normal_vectors_norm, labels.double())
            running_mean += loss_deg_mean.item()
            running_median += loss_deg_median.item()

        # Log Epoch Loss
        num_samples = (len(validationLoader))
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
        print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))
        epoch_mean = running_mean / num_samples
        epoch_median = running_median / num_samples
        print('Val Epoch Mean: {:.4f}'.format(epoch_mean))
        print('Val Epoch Median: {:.4f}'.format(epoch_median))
        writer.add_scalar('data/Val Epoch Mean Error (deg)', epoch_mean, total_iter_num)
        writer.add_scalar('data/Val Epoch Median Error (deg)', epoch_median, total_iter_num)

        # Log 10 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            if config.train.model == 'refinenet':
                output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                       int(normal_vectors_norm.shape[3] * 4))
            elif config.train.model == 'densenet':
                output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                       int(normal_vectors_norm.shape[3] * 2))
            else:
                output = normal_vectors_norm.detach().cpu()
            grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                                 output.float(),
                                                 labels.detach().cpu(),
                                                 max_num_images_to_save=10)
            writer.add_image('Validation', grid_image, total_iter_num)

    ###################### Test Cycle - Real #############################
    if db_test_list:
        print('\nTesting:')
        print('=' * 10)

        model.eval()

        running_mean = 0
        running_median = 0
        running_d1 = 0
        img_tensor_list = []
        output_tensor_list = []
        label_tensor_list = []
        for iter_num, sample_batched in enumerate(tqdm(testLoader)):
            inputs, labels, masks = sample_batched

            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)

            # Save output images, one at a time, to results
            img_tensor = inputs.detach().cpu()
            output_tensor = normal_vectors_norm.detach().cpu()
            label_tensor = labels.detach().cpu()
            mask_tensor = masks.squeeze(1)

            img_tensor_list.append(img_tensor)
            output_tensor_list.append(output_tensor)
            label_tensor_list.append(label_tensor)

            for iii, sample_batched in enumerate(zip(img_tensor, output_tensor, label_tensor, mask_tensor)):
                img, output, label, mask = sample_batched
                # Calc metrics
                loss_deg_mean, loss_deg_median, percentage_1, _, _, _ = loss_functions.metric_calculator(output,
                                                                                                         label,
                                                                                                         mask=mask)
                running_mean += loss_deg_mean.item()
                running_median += loss_deg_median.item()
                running_d1 += percentage_1.item()

            # statistics
            running_loss += loss.item()
            writer.add_scalar('data/Test Real Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
            writer.add_scalar('data/Test Real Median Error (deg)', loss_deg_median.item(), total_iter_num)

        # Log Epoch Loss
        num_samples = len(testLoader.dataset)
        epoch_mean = running_mean / num_samples
        epoch_median = running_median / num_samples
        epoch_d1 = running_d1 / num_samples
        print('Test Real Epoch Mean: {:.4f}'.format(epoch_mean))
        print('Test Real Epoch Median: {:.4f}'.format(epoch_median))
        writer.add_scalar('data/Test Real Epoch Mean Error (deg)', epoch_mean, total_iter_num)
        writer.add_scalar('data/Test Real Epoch Median Error (deg)', epoch_median, total_iter_num)
        writer.add_scalar('data/Test Real Epoch d1 Error (deg)', epoch_d1, total_iter_num)

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            grid_image = utils.create_grid_image(torch.cat(img_tensor_list, dim=0),
                                                 torch.cat(output_tensor_list, dim=0),
                                                 torch.cat(label_tensor_list, dim=0),
                                                 max_num_images_to_save=200)
            writer.add_image('Test Real', grid_image, total_iter_num)

    ###################### Test Cycle - Synthetic #############################
    if db_test_synthetic_list:
        print('\nTest Synthetic:')
        print('=' * 10)

        model.eval()

        running_loss = 0.0
        running_mean = 0
        running_median = 0
        for iter_num, sample_batched in enumerate(tqdm(testSyntheticLoader)):
            inputs, labels, masks = sample_batched

            # Forward pass of the mini-batch
            if config.train.model == 'refinenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 4), int(labels.shape[3] / 4))
                labels_resized = labels_resized.to(device)
            if config.train.model == 'densenet':
                labels_resized = resize_tensor(labels, int(labels.shape[2] / 2), int(labels.shape[3] / 2))
                labels_resized = labels_resized.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                normal_vectors = model(inputs)

            normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

            if config.train.model == 'unet':
                loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
            elif config.train.model == 'deeplab_xception' or config.train.model == 'deeplab_resnet' or config.train.model == 'drn':
                loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
                loss /= config.train.batchSize
            elif config.train.model == 'refinenet' or config.train.model == 'densenet':
                # refinenet gives output that is 1/4th the size of input, hence resize labels to match the output size of refinenet
                loss = criterion(normal_vectors_norm, labels_resized.double(), reduction='sum')
                loss /= config.train.batchSize

            running_loss += loss.item()

            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
                normal_vectors_norm, labels.double())
            running_mean += loss_deg_mean.item()
            running_median += loss_deg_median.item()

        # Log Epoch Loss
        num_samples = (len(testSyntheticLoader))
        epoch_loss = running_loss / num_samples
        writer.add_scalar('data/Test Synthetic Epoch Loss', epoch_loss, total_iter_num)
        print('\Test Synthetic Epoch Loss: {:.4f}'.format(epoch_loss))
        epoch_mean = running_mean / num_samples
        epoch_median = running_median / num_samples
        print('Test Synthetic Epoch Mean: {:.4f}'.format(epoch_mean))
        print('Test Synthetic Epoch Median: {:.4f}'.format(epoch_median))
        writer.add_scalar('data/Test Synthetic Epoch Mean Error (deg)', epoch_mean, total_iter_num)
        writer.add_scalar('data/Test Synthetic Epoch Median Error (deg)', epoch_median, total_iter_num)

        # Log 30 images every N epochs
        if (epoch % config.train.saveImageInterval) == 0:
            if config.train.model == 'refinenet':
                output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 4),
                                       int(normal_vectors_norm.shape[3] * 4))
            elif config.train.model == 'densenet':
                output = resize_tensor(normal_vectors_norm.detach().cpu(), int(normal_vectors_norm.shape[2] * 2),
                                       int(normal_vectors_norm.shape[3] * 2))
            else:
                output = normal_vectors_norm.detach().cpu().float()

            grid_image = utils.create_grid_image(inputs.detach().cpu(),
                                                 output,
                                                 labels.detach().cpu(),
                                                 max_num_images_to_save=16)
            writer.add_image('Test Synthetic', grid_image, total_iter_num)

writer.close()
