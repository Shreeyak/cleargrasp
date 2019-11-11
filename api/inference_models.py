import os

import cv2
import imgaug as ia
import numpy as np
import scipy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa

from . import utils
from .modeling import deeplab, deeplab_masks


class InferenceOutlines():

    def __init__(self, outlinesWeightsFile, outlinesModel='deeplab_resnet', imgHeight=288, imgWidth=512):
        '''Class to run Inference of the Outlines Prediction Model

        Args:
            outlinesWeightsFile (str): Path to the weights file for the model
            outlinesModel (str): Which model to use. Can be one of ["unet", "deeplab_resnet", "drn"]
            imgHeight (int): The height to which images should be resized to before passing to the model.
            imgWidth (int): The width to which images should be resized to before passing to the model.
        '''

        # Create Model
        if outlinesModel == 'deeplab_resnet':
            print('Creating Deeplabv3-Resnet model for outlines and loading checkpoint')
            self.model = deeplab_masks.DeepLab(num_classes=3, backbone='resnet', sync_bn=True,
                            freeze_bn=False)
        elif outlinesModel == 'drn':
            print('Creating DRN model for outlines and loading checkpoint')
            self.model = deeplab_masks.DeepLab(num_classes=3, backbone='drn', sync_bn=True,
                                               freeze_bn=True)  # output stride is 8 for drn
        else:
            raise NotImplementedError(
                'Model may only be "drn", "deeplab_resnet" or unet". Given model is: {}'.format(outlinesModel))

        # Load Checkpoint and its data (weights file)
        if not os.path.isfile(outlinesWeightsFile):
            raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
                outlinesWeightsFile))

        checkpoint = torch.load(outlinesWeightsFile, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # Newer checkpoints have multiple dicts within
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create Transforms
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": imgHeight,
                "width": imgWidth
            }, interpolation='nearest'),
        ])

    def runOnNumpyImage(self, img, scaling_factor_weights=3):
        '''Runs inference of Outlines prediction model on a Numpy Image

        Args:
            img (numpy.ndarray): shape=(H, W, 3)
                                 if dtype=np.float32, Expected data in range (0, 1)
                                 if dtype=np.uint8, Expected data in range (0, 255)
                                 Obtained normally by using PIL to open image (in mode RGB) and converting to numpy.
            scaling_factor_weights (int): The power to which outputs of outlines are raised to before scaling to
                                          range 0-1000. The default value, as used by Yinda, is 3.

        Returns:
            numpy.ndarray: Occlusion Weights derived from outlines prediction. Used by depth2depth module.
                           shape=(H, W), dtype=np.uint16
            numpy.ndarray: RGB Visualization of occlusion Weights.
                           shape=(H, W, 3), dtype=np.uint8
            numpy.ndarray: Predicted Outlines RGB image. Value of each color corresponds to the class it belongs to:
                                Red: Background, Green: Depth Boundary, Blue: Surface Normal Gradient
                           shape=(H, W, 3), dtype=np.uint8.
        '''
        with torch.no_grad():
            # Resize Image and convert to tensor
            det_tf = self.transform.to_deterministic()
            img = det_tf.augment_image(img)
            inputs = transforms.ToTensor()(img)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.to(self.device)

            outputs = self.model(inputs)
            predictions = torch.max(outputs, 1)[1]

            # Generate and Save Occlusion Weights File used by depth2depth
            # calculating occlusion weights
            SCALING_FACTOR_PWR = scaling_factor_weights
            SCALING_FACTOR_MUL = 1000
            output_softmax = nn.Softmax(dim=1)(outputs).squeeze(0).cpu().numpy()
            # TEST - REMOVING UNCERTAIN VALUES FROM OUTLINE WEIGHTS
            # output_softmax[output_softmax < 0.3] = 1e-4
            weight = (1 - output_softmax[1, :, :])  # Occlusion (depth) boundaries is channel 1
            x = np.power(weight, SCALING_FACTOR_PWR)
            x = np.multiply(x, SCALING_FACTOR_MUL)
            occlusion_boundary_weight = x.astype(np.uint16)
            # Increase the min and max values by small amount epsilon so that absolute min/max values
            # don't cause problems in the depth2depth optimization code.
            epsilon = 1
            occlusion_boundary_weight[occlusion_boundary_weight == 0] += epsilon
            occlusion_boundary_weight[occlusion_boundary_weight == SCALING_FACTOR_MUL] -= epsilon

            # Convert prediction to RGB for display (R: background, G: occlusion boundary, B: normals gradient)
            predictions = predictions.squeeze(0).cpu().numpy()
            output_rgb = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
            output_rgb[:, :, 0][predictions == 0] = 255
            output_rgb[:, :, 1][predictions == 1] = 255
            output_rgb[:, :, 2][predictions == 2] = 255

            # Create RGB Visualization of Occlusion Weights
            NORM_FACTOR_UINT8 = SCALING_FACTOR_MUL / 255.0
            occlusion_boundary_weight_rgb = (occlusion_boundary_weight / NORM_FACTOR_UINT8).astype(np.uint8)
            occlusion_boundary_weight_rgb = cv2.applyColorMap(occlusion_boundary_weight_rgb, cv2.COLORMAP_OCEAN)
            occlusion_boundary_weight_rgb = cv2.cvtColor(occlusion_boundary_weight_rgb, cv2.COLOR_BGR2RGB)

        return occlusion_boundary_weight, occlusion_boundary_weight_rgb, output_rgb


class InferenceNormals():

    def __init__(self, normalsWeightsFile, normalsModel='deeplab_resnet', imgHeight=288, imgWidth=512):
        '''Class to run Inference of the normals Prediction Model

        Args:
            normalsWeightsFile (str): Path to the weights file for the model
            normalsModel (str): Which model to use. Can be one of ["unet", "deeplab_resnet", "drn"]
            imgHeight (int): The height to which images should be resized to before passing to the model.
            imgWidth (int): The width to which images should be resized to before passing to the model.
        '''

        # Create Model
        if normalsModel == 'deeplab_resnet':
            print('Creating Deeplabv3-Resnet model for surface normals and loading checkpoint')
            self.model = deeplab.DeepLab(num_classes=3, backbone='resnet', sync_bn=True,
                                         freeze_bn=False)
        elif normalsModel == 'drn':
            print('Creating DRN model for normals and loading checkpoint')
            self.model = deeplab.DeepLab(num_classes=3, backbone='drn', sync_bn=True,
                                         freeze_bn=False)  # output stride is 8 for drn
        else:
            raise NotImplementedError(
                'Model may only be "unet" or "deeplab_resnet". Given model is: {}'.format(normalsModel))

        # Load Checkpoint and its data (weights file)
        if not os.path.isfile(normalsWeightsFile):
            raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
                normalsWeightsFile))

        checkpoint = torch.load(normalsWeightsFile, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # Newer checkpoints have multiple dicts within
            # checkpoint['model_state_dict'].pop('decoder.last_conv.8.weight')
            # checkpoint['model_state_dict'].pop('decoder.last_conv.8.bias')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            checkpoint['state_dict'].pop('decoder.last_conv.8.weight')
            checkpoint['state_dict'].pop('decoder.last_conv.8.bias')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create Transforms
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": imgHeight,
                "width": imgWidth
            }, interpolation='nearest'),
        ])

    def _normal_to_rgb(self, normals_to_convert):
        '''Converts a surface normals array into an RGB image.
        Surface normals are represented in a range of (-1,1),
        This is converted to a range of (0,255) to be written
        into an image.
        The surface normals are normally in camera co-ords,
        with positive z axis coming out of the page. And the axes are
        mapped as (x,y,z) -> (R,G,B).
        '''
        camera_normal_rgb = normals_to_convert + 1
        camera_normal_rgb *= 127.5
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
        return camera_normal_rgb

    def _convert_normals_depth2depth_format(self, normals):
        """This function converts surface normals from our representation to the representation used by depth2depth

        When displaying normals, RGB represents the axes XYZ. Using that demonination, here are the representations
        used by our method and theirs:
          - Ours:        R: Right,  G: Up,               B: Towards Camera  (Y-up notation)
          - depth2depth: R: Right,  G: Away from Camera, B: Up              (Z-up notation)
        Args:
            normals (numpy.ndarray): Surface normals. Shape=(3, H, W), dtype=float32
        """
        if (len(normals.shape) != 3) or (normals.shape[0] != 3):
            raise ValueError('Shape of normals should be (3, H, W). Got shape: {}'.format(normals.shape))

        # Convert normals to shape (N, 3)
        normals_list = np.reshape(normals, (3, -1)).transpose((1, 0))

        # Apply Rotation
        r = scipy.spatial.transform.Rotation.from_euler('x', 90, degrees=True)
        normals_list_rotated = r.apply(normals_list)

        # Convert normals back to shape (3, H, W)
        normals_depth2depth_format = np.reshape(normals_list_rotated.transpose(1, 0), normals.shape)

        return normals_depth2depth_format

    def runOnNumpyImage(self, img):
        '''Runs inference of Outlines prediction model on a Numpy Image

        Args:
            img (numpy.ndarray): shape=(H, W, 3)
                                 if dtype=np.float32, Expected data in range (0, 1)
                                 if dtype=np.uint8, Expected data in range (0, 255)
                                 Obtained normally by using PIL to open image (in mode RGB) and converting to numpy.

        Returns:
            numpy.ndarray: Predicted Surface Normals
                           shape=(3, H, W), dtype=np.float32 with range (-1, 1)
            numpy.ndarray: RGB representation of predicted Surface Normals. (R, G, B) represent (X, Y, Z) axes.
        '''
        with torch.no_grad():
            # Resize Image and convert to tensor
            det_tf = self.transform.to_deterministic()
            img = det_tf.augment_image(img)
            inputs = transforms.ToTensor()(img)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.to(self.device)

            outputs = self.model(inputs)

            outputs_norm = nn.functional.normalize(outputs, p=2, dim=1)
            surface_normals = outputs_norm.squeeze(0).cpu().numpy()

            # Create RGB Viz of Normals
            surface_normals_rgb = self._normal_to_rgb(surface_normals.transpose((1, 2, 0)))

            # Rotate Surface Normals into depth2depth notation
            surface_normals_rotated = self._convert_normals_depth2depth_format(surface_normals)

            # # Create RGB Viz of Rotated Normals
            # surface_normals_rgb = self._normal_to_rgb(surface_normals_rotated.transpose((1, 2, 0)))

        return surface_normals_rotated, surface_normals_rgb


class InferenceMasks():

    def __init__(self, masksWeightsFile, masksModel='drn', imgHeight=288, imgWidth=512):
        '''Class to run Inference of the Outlines Prediction Model

        Args:
            masksWeightsFile (str): Path to the weights file for the model
            masksModel (str): Which model to use. Can be one of ["deeplab_xception", "deeplab_resnet", "drn"]
            imgHeight (int): The height to which images should be resized to before passing to the model.
            imgWidth (int): The width to which images should be resized to before passing to the model.
        '''

        # Create Model
        if masksModel == 'deeplab_resnet':
            print('Creating Deeplabv3-Resnet model for masks and loading checkpoint')
            self.model = deeplab_masks.DeepLab(num_classes=2, backbone='resnet', sync_bn=True,
                                               freeze_bn=False)
        elif masksModel == 'deeplab_xception':
            self.model = deeplab_masks.DeepLab(num_classes=2, backbone='xception', sync_bn=True,
                                               freeze_bn=False)
        elif masksModel == 'drn':
            print('Creating DRN model for masks and loading checkpoint')
            self.model = deeplab_masks.DeepLab(num_classes=2, backbone='drn', sync_bn=True,
                                               freeze_bn=False)  # output stride is 8 for drn
        else:
            raise NotImplementedError(
                'Model may only be "unet" or "deeplab_resnet". Given model is: {}'.format(masksModel))

        # Load Checkpoint and its data (weights file)
        if not os.path.isfile(masksWeightsFile):
            raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
                masksWeightsFile))

        checkpoint = torch.load(masksWeightsFile, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # Newer checkpoints have multiple dicts within
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        elif 'state_dict' in checkpoint:
            checkpoint['state_dict'].pop('decoder.last_conv.8.weight')
            checkpoint['state_dict'].pop('decoder.last_conv.8.bias')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise RuntimeError('Invalid Checkpoint. It does not contain "model_state_dict" in it:\n{}'.format(
                checkpoint.keys()))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create Transforms
        self.transform = iaa.Sequential([
            iaa.Resize({
                "height": imgHeight,
                "width": imgWidth
            }, interpolation='nearest'),
        ])

    def runOnNumpyImage(self, img, dilate_mask=False):
        '''Runs inference of Outlines prediction model on a Numpy Image

        Args:
            img (numpy.ndarray): shape=(H, W, 3)
                                 if dtype=np.float32, Expected data in range (0, 1)
                                 if dtype=np.uint8, Expected data in range (0, 255)
                                 Obtained normally by using PIL to open image (in mode RGB) and converting to numpy.

        Returns:
            numpy.ndarray: Binary mask of detected transparent objects. Used to mask input depth.
                           shape=(H, W), dtype=np.uint8, True pixels have value of 255
        '''
        with torch.no_grad():
            # Resize Image and convert to tensor
            det_tf = self.transform.to_deterministic()
            img = det_tf.augment_image(img)
            inputs = transforms.ToTensor()(img)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.to(self.device)

            outputs = self.model(inputs)
            predictions = torch.max(outputs, 1)[1]
            predictions = predictions.squeeze(0).cpu().numpy()

            mask = np.zeros(predictions.shape, dtype=np.uint8)
            mask[predictions == 1] = 255

            if dilate_mask is True:
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

        return mask