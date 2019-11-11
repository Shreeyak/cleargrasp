'''Functions for reading and saving EXR images using OpenEXR.
'''

import sys

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn

sys.path.append('../..')
from api import utils as api_utils
from api.utils import exr_loader, exr_saver


def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    return camera_normal_rgb


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]

    output_tensor = outputs[:max_num_images_to_save]
    output_tensor_rgb = normal_to_rgb(output_tensor)

    label_tensor = labels[:max_num_images_to_save]
    mask_invalid_pixels = torch.all(label_tensor == 0, dim=1, keepdim=True)
    mask_invalid_pixels = (torch.cat([mask_invalid_pixels] * 3, dim=1)).byte()

    label_tensor_rgb = normal_to_rgb(label_tensor)
    label_tensor_rgb[mask_invalid_pixels] = 0

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    x = cos(output_tensor, label_tensor)
    # loss_cos = 1.0 - x
    loss_rad = torch.acos(x)

    loss_rad_rgb = np.zeros((loss_rad.shape[0], 3, loss_rad.shape[1], loss_rad.shape[2]), dtype=np.float32)
    for idx, img in enumerate(loss_rad.numpy()):
        error_rgb = api_utils.depth2rgb(img,
                                        min_depth=0.0,
                                        max_depth=1.57,
                                        color_mode=cv2.COLORMAP_PLASMA,
                                        reverse_scale=False)
        loss_rad_rgb[idx] = error_rgb.transpose(2, 0, 1) / 255
    loss_rad_rgb = torch.from_numpy(loss_rad_rgb)
    loss_rad_rgb[mask_invalid_pixels] = 0

    mask_invalid_pixels_rgb = torch.ones_like(img_tensor)
    mask_invalid_pixels_rgb[mask_invalid_pixels] = 0

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb, loss_rad_rgb, mask_invalid_pixels_rgb), dim=3)
    # grid_image = make_grid(images, 1, normalize=True, scale_each=True)
    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter)**power)
