import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import cv2

def label_to_rgb(label):
    '''Output RGB visualizations of the outlines' labels

    The labels of outlines have 3 classes: Background, Depth Outlines, Surface Normal Outlines which are mapped to
    Red, Green and Blue respectively.

    Args:
        label (numpy.ndarray): Shape: (height, width). Each pixel contains an int with value of class.

    Returns:
        numpy.ndarray: Shape (height, width, 3): RGB representation of the labels
    '''
    rgbArray = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    rgbArray[:, :, 0][label == 0] = 255
    rgbArray[:, :, 1][label == 1] = 255
    rgbArray[:, :, 2][label == 2] = 255

    return rgbArray

masks_outlines = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/hearts/sources/hearts-in-containers-val/source-files/outlines-from-masks/rgb-visualizations/000000003-s-outlineSegmentationRgb.png'
outlines_without_touching_surface = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/hearts/sources/hearts-in-containers-val/source-files/outlines/rgb-visualizations/000000003-outlineSegmentationRgb.png'
outlines_without_touching_surface_png = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/hearts/sources/hearts-in-containers-val/source-files/outlines/000000003-outlineSegmentation.png'

outlines_from_masks = imageio.imread(masks_outlines)
outlines_from_masks = outlines_from_masks[:, :, 1]

outlines_from_depth = imageio.imread(outlines_without_touching_surface)
outlines_from_depth = outlines_from_depth[:, :, 1]
kernel = np.ones((3, 3), np.uint8)
outlines_from_depth = cv2.dilate(outlines_from_depth, kernel, iterations=2)


outlines_touching_floor = outlines_from_masks - outlines_from_depth
outlines_touching_floor = cv2.erode(outlines_touching_floor, kernel, iterations=2)
outlines_touching_floor = cv2.dilate(outlines_touching_floor, kernel, iterations=3)
# outlines_touching_floor = cv2.dilate(outlines_touching_floor, kernel, iterations=2)
# outlines_touching_floor = cv2.dilate(outlines_touching_floor, kernel, iterations=2)


result = imageio.imread(outlines_without_touching_surface_png)
result[result == 2] = 0
result[outlines_touching_floor == 255] = 2
result = label_to_rgb(result)


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax1.title.set_text('outlines_from_masks')
ax2.title.set_text('outlines_from_depth')
ax3.title.set_text('outlines_touching_floor')
ax4.title.set_text('result')

ax1.imshow(outlines_from_masks, aspect='auto')
ax2.imshow(outlines_from_depth, aspect='auto')
ax3.imshow(outlines_touching_floor, aspect='auto')
ax4.imshow(result, aspect='auto')
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()

# output_color = label_to_rgb(output)
# imageio.imwrite('/home/gani/deeplearning/brain/datasets/test-set-paper/val/1.png', outlines_required)
















# shrek -------------------------------------------------------------------------------------------------------------------
# mask_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/hearts/sources/hearts-in-containers-val/source-files/segmentation-masks/000000003-segmentation-mask.png'
# outline_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/hearts/sources/hearts-in-containers-val/source-files/thres-8-outlines/rgb-visualizations/000000003-outlineSegmentationRgb.png'
# result_file = '/home/gani/deeplearning/brain/datasets/test-set-paper/val/1.png'

# mask = imageio.imread(mask_file)
# outline = imageio.imread(outline_file)

# # Sobel
# kernel_size = 3
# threshold = 1.0

# sobelx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=kernel_size)
# sobely = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=kernel_size)
# sobelx = np.abs(sobelx)
# sobely = np.abs(sobely)

# sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
# sobelx_binary[sobelx >= threshold] = True

# sobely_binary = np.full(sobely.shape, False, dtype=bool)
# sobely_binary[sobely >= threshold] = True

# sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

# sobel_result = np.zeros_like(depth_img_orig, dtype=np.uint8)
# sobel_result[sobel_binary] = 255


