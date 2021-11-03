import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

# from skimage.transform import resize
from skimage.transform import resize

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import OpenEXR
import Imath
from scipy.misc import imsave
import imageio
from PIL import Image

from utils import exr_loader


class OPT():
    def __init__(self):
        self.dataroot = './data/'
        self.file_list = './data/datalist'
        self.batchSize = 32
        self.shuffle = True
        self.phase = 'train'
        self.num_epochs = 500
        self.imsize = 224
        self.num_classes = int(3)
        self.gpu = '0'
        self.logs_path = 'logs/exp9'
        self.use_pretrained = False


opt = OPT()

depth_path = '/home/gani/shrek-hdd/datasets/google-brain/transparent-objects/milk-bottles/complete-set/milk-bottles/source-files/depth-imgs/%09d-depth.exr'
path_save_depth_edges = './data/edges-depth-imgs/%09d-depth-edges.png'
normals_path = '/home/gani/shrek-hdd/datasets/google-brain/transparent-objects/milk-bottles/complete-set/milk-bottles/source-files/camera-normals/%09d-cameraNormals.exr'
path_save_normal_edges = './data/edges-normals-imgs/%09d-normals-edges.png'
path_save_combined_outline = './data/samples/combined_edges/%09d-segmentation.png'
path_save_combined_outline_viz = './data/samples/combined_edges/viz/%09d-rgb.png'
# depth_mask = './data/combined-edges/depth-mask/%09d-rgb.jpg'
allchannels = []
empty_channel = np.zeros((1080, 1920), 'uint8')
height = 1080
width = 1920


def label_to_rgb(label):
    '''Output RGB visualizations of the labels (outlines)
    Assumes labels have int values and max number of classes = 3

    Args:
        label (numpy.ndarray): Shape (height, width). Each pixel contains an int with value of class that it belongs to.

    Returns:
        numpy.ndarray: Shape (height, width, 3): RGB representation of the labels
    '''
    rgbArray = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    rgbArray[:, :, 0][label == 0] = 255
    rgbArray[:, :, 1][label == 1] = 255
    rgbArray[:, :, 2][label == 2] = 255

    return rgbArray

def outline_from_depth(depth_img_orig):
    kernel_size = 9
    threshold = 10
    max_depth_to_object = 2.5

    # Apply Laplacian filters for edge detection for depth images
    depth_img_blur = cv2.GaussianBlur(depth_img_orig, (5, 5), 0)

    # Make all depth values greater than 2.5m as 0 (for masking edge matrix)
    depth_img_mask = depth_img_blur.copy()
    depth_img_mask[depth_img_mask > 2.5] = 0
    depth_img_mask[depth_img_mask > 0] = 1

    # Apply Laplacian filters for edge detection
    # Laplacian Parameters
    edges_lap = cv2.Laplacian(depth_img_orig, cv2.CV_64F, ksize=kernel_size, borderType=0)
    edges_lap = (np.absolute(edges_lap).astype(np.uint8))

    edges_lap_binary = np.zeros(edges_lap.shape, dtype=np.uint8)
    edges_lap_binary[edges_lap > threshold] = 255
    edges_lap_binary[depth_img_orig > max_depth_to_object] = 0

    # edges_lap = cv2.Laplacian(depth_img_blur, cv2.CV_64F, ksize=7, borderType=0 )
    # edges_lap = np.absolute(edges_lap).astype(np.uint8)

    # # convert to binary and apply mask
    # depth_edges = np.zeros(depth_img_orig.shape, dtype = np.uint8)  # edges_lap.copy()
    # depth_edges[edges_lap>1] = 255
    # depth_edges[edges_lap<=1] = 0

    # # Make all depth values greater than 2.5m as 0 (for masking gradients near horizon)
    # max_distance_to_object = 2.5
    # depth_edges[ depth_img_orig > max_distance_to_object] = 0

    return edges_lap_binary


def outline_from_normal(surface_normal):
    ''' surface normal shape = 3 * H * W
    '''
    surface_normal = (surface_normal + 1) / 2  # convert to [0,1] range

    surface_normal_rgb16 = (surface_normal * 65535).astype(np.uint16)
    # surface_normal_rgb8 = (surface_normal * 255).astype(np.uint8).transpose((1,2,0))

    # Take each channel of RGB image one by one, apply gradient and combine
    sobelxy_list = []
    for surface_normal_gray in surface_normal_rgb16:
        # Sobel Filter Params
        # These params were chosen using trial and error.
        # NOTE!!!! The max value of sobel output increases exponentially with increase in kernel size.
        # Print the min/max values of array below to get an idea of the range of values in Sobel output.
        kernel_size = 5
        threshold = 60000

        # Apply Sobel Filter
        sobelx = cv2.Sobel(surface_normal_gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(surface_normal_gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
#         print('\ntype0', sobelx.dtype, sobely.dtype)
#         print('min', np.amin(sobelx), np.amin(sobely))
#         print('max', np.amax(sobelx), np.amax(sobely))

        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)
#         print('\ntype1', sobelx.dtype, sobely.dtype)
#         print('min', np.amin(sobelx), np.amin(sobely))
#         print('max', np.amax(sobelx), np.amax(sobely))

        # Convert to binary
        sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
        sobelx_binary[sobelx >= threshold] = True

        sobely_binary = np.full(sobely.shape, False, dtype=bool)
        sobely_binary[sobely >= threshold] = True

        sobelxy_binary = np.logical_or(sobelx_binary, sobely_binary)
        sobelxy_list.append(sobelxy_binary)

    sobelxy_binary3d = np.array(sobelxy_list).transpose((1, 2, 0))
    sobelxy_binary3d = sobelxy_binary3d.astype(np.uint8) * 255

    sobelxy_binary = np.zeros((surface_normal_rgb16.shape[1], surface_normal_rgb16.shape[2]))
    for channel in sobelxy_list:
        sobelxy_binary[channel > 0] = 255

    # print('normal nonzero:', np.sum((edges_sobel_binary > 0) & (edges_sobel_binary < 255)))
    return sobelxy_binary


for i in range(100, 2751):
    # Load Depth Img convert to outlines and resize
    print('Loading img %d' % (i))
    depth_img_orig = exr_loader(depth_path % (i), ndim=1)
    depth_edges = outline_from_depth(depth_img_orig)
    depth_edges_img = Image.fromarray(depth_edges, 'L').resize((width, height), resample=Image.NEAREST)

    depth_edges = np.asarray(depth_edges_img)

    # Load RGB image, convert to outlines and  resize
    surface_normal = exr_loader(normals_path % (i))
    normals_edges = outline_from_normal(surface_normal)
    # edges = Image.fromarray(edges).resize((224,224))

    save_output = True
    if(save_output):
        depth_edges_img.save(path_save_depth_edges % (i))
        imsave(path_save_normal_edges % (i), normals_edges)

    # Depth and Normal outlines should not overlap. Priority given to depth.
    depth_edges = depth_edges.astype(np.uint8)
    normals_edges[depth_edges == 255] = 0

    # modified edges and create mask
    output = np.zeros((height, width), 'uint8')
    output[normals_edges == 255] = 2
    output[depth_edges == 255] = 1

    # Remove gradient bars from the top and bottom of img
    num_of_rows_to_delete = 2
    output[:num_of_rows_to_delete, :] = 0
    output[-num_of_rows_to_delete:, :] = 0

    img = Image.fromarray(output, 'L')
    img.save(path_save_combined_outline % i)

    # visualization of outline
    # rgbArray0 = np.zeros((height, width), 'uint8')
    # rgbArray1 = np.zeros((height, width), 'uint8')
    # rgbArray2 = np.zeros((height, width), 'uint8')
    # rgbArray0[output == 0] = 255
    # rgbArray1[output == 1] = 255
    # rgbArray2[output == 2] = 255
    # rgbArray = np.stack((rgbArray0, rgbArray1, rgbArray2), axis=2)
    output_color = label_to_rgb(output)

    img = Image.fromarray(output_color, 'RGB')
    img.save(path_save_combined_outline_viz % i)


# print(allchannels)
'''
    display_output = 1
    if(display_output):
        fig1 = plt.figure(figsize=(12,12))
        plt.imshow(depth_img_orig, cmap='gray')
        plt.show()
        fig1 = plt.figure(figsize=(12,12))
        plt.imshow(depth_img_blur, cmap='gray')
        plt.show()
        fig2 = plt.figure(figsize=(12,12))
        plt.imshow(edges_lap, cmap='gray')
        plt.show()
        fig3 = plt.figure(figsize=(12,12))
        plt.imshow(depth_edges, cmap='gray')
        plt.show()
        fig4 = plt.figure(figsize=(12,12))
        plt.imshow(edges, cmap='gray')
        plt.show()
'''
