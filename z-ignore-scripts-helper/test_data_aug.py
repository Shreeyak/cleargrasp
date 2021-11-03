"""Runs data augmentation through a dataloader+imgaug on a dataset and saves a collage
of the results to provide an idea of output of data augmentation.
"""
import argparse
import errno
import glob
import os

import imageio
import imgaug as ia
import numpy as np
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


class RgbDataset(Dataset):
    """
    Dataset class for returning only RGB images for testing data augmentation outputs.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
    """

    def __init__(
            self,
            input_dir,
            transform=None,
    ):
        super().__init__()
        self.images_dir = input_dir
        self.transform = transform

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._extension_input = ['-transparent-rgb-img.jpg', '-rgb.jpg']  # The file extension of input images
        self._create_lists_filenames(self.images_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
        '''
        image_path = self._datalist_input[index]
        _img = imageio.imread(image_path)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)

        _img_tensor = transforms.ToTensor()(_img)
        return _img_tensor

    def _create_lists_filenames(self, images_dir):
        '''Creates a list of filenames of images in given dir.

        Args:
            images_dir (str): Path to the dir where images are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
        '''

        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        for ext in self._extension_input:
            imagepaths = sorted(glob.glob(os.path.join(images_dir, '*' + ext)))
            self._datalist_input = self._datalist_input + imagepaths

        assert len(self._datalist_input) > 0, 'No images found in given directory: {}'.format(images_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create collage of outputs of data augmentation')

    parser.add_argument('-s',
                        '--source_dir',
                        required=True,
                        type=str,
                        help='Path to dir of RGB images',
                        metavar='path/to/dataset')
    parser.add_argument('-d',
                        '--dest_dir',
                        required=True,
                        type=str,
                        help='Path to store results',
                        metavar='path/to/dataset')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for dataloader')
    args = parser.parse_args()

    # Create directory to save results
    results_root_dir = args.dest_dir
    runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(results_dir):
        NUM_FILES_IN_EMPTY_FOLDER = 0
        if len(os.listdir(results_dir)) > NUM_FILES_IN_EMPTY_FOLDER:
            prev_run_id += 1
            results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)

    testBatchSize = args.batch_size
    results_store_dir = results_dir
    IMSIZE = 256

    augs_test = iaa.Sequential([
        # Geometric Augs
        iaa.Resize({
            "height": IMSIZE,
            "width": IMSIZE
        }, interpolation='nearest'),

        # Bright Patches
        iaa.Sometimes(
            1.0,
            iaa.blend.Alpha(factor=(0.2, 0.7),
                            first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                              upscale_method='cubic',
                                                              iterations=(1, 2)),
                            name="simplex-blend")),

        # Color Space Mods
        iaa.Sometimes(
            1.0,
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
            1.0,
            iaa.SomeOf((1, None), [
                iaa.OneOf(
                    [iaa.MotionBlur(k=3, name="motion-blur"),
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
        iaa.Sometimes(1.0, iaa.CoarseDropout(0.03, size_px=(4, 8), per_channel=True, name="cdropout")),
    ])
    input_only = [
        "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
        "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
    ]

    db_test = RgbDataset(input_dir=args.source_dir, transform=augs_test)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    for ii, sample_batched in enumerate(testloader):
        inputs = sample_batched

        imgs_per_row = 6
        img_grid = make_grid(inputs, nrow=imgs_per_row, padding=2)
        save_image(img_grid, os.path.join(results_store_dir, '{:09d}-results.png'.format(ii)))

        print('  Saved batch {:05d}'.format(ii))
