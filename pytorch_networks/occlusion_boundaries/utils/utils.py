'''Contains utility functions used by train/eval code.
'''
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np


def label_to_rgb(label):
    '''Output RGB visualizations of the outlines' labels

    The labels of outlines have 3 classes: Background, Depth Outlines, Surface Normal Outlines which are mapped to
    Red, Green and Blue respectively.

    Args:
        label (torch.Tensor): Shape: (batchSize, 1, height, width). Each pixel contains an int with value of class.

    Returns:
        torch.Tensor: Shape (no. of images, 3, height, width): RGB representation of the labels
    '''
    if len(label.shape) == 4:
        # Shape: (batchSize, 1, height, width)
        rgbArray = torch.zeros((label.shape[0], 3, label.shape[2], label.shape[3]), dtype=torch.float)
        rgbArray[:, 0, :, :][label[:, 0, :, :] == 0] = 1
        rgbArray[:, 1, :, :][label[:, 0, :, :] == 1] = 1
        rgbArray[:, 2, :, :][label[:, 0, :, :] == 2] = 1
    if len(label.shape) == 3:
        # Shape: (1, height, width)
        rgbArray = torch.zeros((3, label.shape[1], label.shape[2]), dtype=torch.float)
        rgbArray[0, :, :][label[0, :, :] == 0] = 1
        rgbArray[1, :, :][label[0, :, :] == 1] = 1
        rgbArray[2, :, :][label[0, :, :] == 2] = 1

    return rgbArray


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]
    output_tensor = torch.unsqueeze(torch.max(outputs[:max_num_images_to_save], 1)[1].float(), 1)
    output_tensor_rgb = label_to_rgb(output_tensor)
    label_tensor = labels[:max_num_images_to_save]
    label_tensor_rgb = label_to_rgb(label_tensor)

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
    grid_image = make_grid(images, 1, normalize=True, scale_each=True)

    return grid_image


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

        `(minibatch, C, d_1, d_2, ..., d_K)`

    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.

    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss


def FocalLoss(logit, target, weight, gamma=2, alpha=0.5, ignore_index=255, size_average=True, batch_average=True):
    
    n, c, h, w = logit.shape
    target = target.squeeze(1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                    reduction='sum')

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    if batch_average:
        loss /= n

    return loss

def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    per_class_iou = [0] * n_classes
    num_images_per_class = [0] * n_classes
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        iou_per_class = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

            if union[j] == 0:
                iou_per_class[j] = -1

            else:
                iou_per_class[j] = intersect[j] / union[j]
                # print('IoU for class %d is %f'%(j, iou_per_class[j]))

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        for k in range(n_classes):
            if iou_per_class[k] == -1:
                continue
            else:
                per_class_iou[k] += iou_per_class[k]
                num_images_per_class[k] += 1

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou
    # print('class iou:', per_class_iou)
    # print('images per class:', num_images_per_class)
    return total_iou, per_class_iou, num_images_per_class

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)