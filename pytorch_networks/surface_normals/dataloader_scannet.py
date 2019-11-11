#!/usr/bin/env python3

from __future__ import print_function, division
import os
import glob
from PIL import Image
import Imath
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio

from utils import utils


class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 input_dir='/home/gani/deeplearning/scannet-rgb/scans/',
                 label_dir='/home/gani/deeplearning/scannet_render_normal/',
                 transform=None,
                 input_only=None,
                 ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only
        self.rgb_folder = 'color'
        self.normal_folder = 'mesh_images'

        # Create list of filenames
        self._datalist_input = None  # Variable containing list of all input images filenames in dataset
        self._datalist_label = None  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = '.jpg'  # The file extension of input images
        self._extension_label = '.exr'  # The file extension of labels
        self._create_lists_filenames(self.images_dir, self.labels_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None)
        '''

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        scene_id = image_path.split('/')[-3]
        normal_file_folder = os.path.join(self.labels_dir, scene_id, self.normal_folder)
        normals = []
        for (dirpath, dirnames, filenames) in os.walk(normal_file_folder):
            for filename in filenames:
                if (filename.endswith(".png")):
                    if (filename.split('_')[0]) == image_path.split('/')[-1].split('.')[0].zfill(9):
                        normals.append(os.path.join(dirpath, filename))
        if len(normals) < 3:
            raise ValueError('labels are not present for {}'.format(image_path))
        normals.sort()
        x = imageio.imread(normals[0])
        x = ((x / 32768) - 1)
        y = imageio.imread(normals[1])
        y = ((y / 32768) - 1)
        z = imageio.imread(normals[2])
        z = ((z / 32768) - 1)
        _label = np.stack((x, y, z), axis=0)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            # img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
            if self.labels_dir:
                # NOTE! EXPERIMENTAL - needs to be checked.

                # covert normals into an image of dtype float32 in range [0, 1] from range [-1, 1]
                _label = (_label + 1) / 2
                _label = _label.transpose((1, 2, 0))  # (H, W, 3)

                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

                # covert normals back to range [-1, 1]
                _label = _label.transpose((2, 0, 1))  # (3, H, W)
                _label = (_label * 2) - 1

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label).float()
        else:
            _label_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        fake_mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor, fake_mask_tensor

    def _create_lists_filenames(self, images_dir, label_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''
        rgb_list = []
        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        scene_list = [os.path.join(images_dir, name) for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
        if len(scene_list) == 0:
            raise ValueError('No house id found in given directory. Searched for {}'.format(images_dir))


        # Since the number of normals file are only a small part of the rgb file list, we are iterating over this to get the respective rgb files.
        for scene in scene_list:
            final_normal_dir = os.path.join(label_dir, os.path.basename(scene), self.normal_folder)
            assert os.path.isdir(final_normal_dir), 'Dataloader given images directory that does not exist: "%s"' % (final_normal_dir)
            for (dirpath, dirnames, filenames) in os.walk(final_normal_dir):
                for filename in filenames:
                    if (filename.endswith(".png")):
                        file_num = filename.split('_')[0].lstrip('0')
                        if len(file_num) == 0:
                            file_num = '0'
                        rgb_list.append(os.path.join(images_dir, scene, self.rgb_folder, file_num + '.jpg'))

        # to remove duplicate entries
        rgb_list = list(set(rgb_list))
        self._datalist_input = rgb_list

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = SurfaceNormalsDataset(
        input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
        label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.show()

        break
