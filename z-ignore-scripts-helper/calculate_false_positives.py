# Use the numpy library.
import numpy as np


def compute_metrics(pred, label):
    """Compute metrics like True/False Positive, True/False Negative.`
        MUST HAVE ONLY 2 CLASSES: BACKGROUND, OBJECT.
    Args:
        pred (numpy.ndarray): Prediction, one-hot encoded. Shape: [2, H, W], dtype: uint8
        label (numpy.ndarray): Ground Truth, one-hot encoded. Shape: [H, W], dtype: uint8
    Returns:
        float: IOU, TP, TN, FP, FN
    """
    if len(pred.shape) > 3:
        raise ValueError("pred should have shape [2, H, W], got: {}".format(pred.shape))
    if len(label.shape) > 2:
        raise ValueError("label should have shape [H, W], got: {}".format(label.shape))

    total_pixels = pred.shape[0] * pred.shape[1]

    tp = np.sum(np.logical_and(pred == 1, label > 0))
    tn = np.sum(np.logical_and(pred == 0, label == 0))
    fp = np.sum(np.logical_and(pred == 1, label == 0))
    fn = np.sum(np.logical_and(pred == 0, label > 0))

    if (tp + tn + fp + fn) != total_pixels:
        raise ValueError('The number of total pixels ({}) and sum of tp,fp,tn,fn ({}) is not equal'.format(
            total_pixels, (tp + tn + fp + fn)))
    iou = tp / (tp + fp + fn)

    _tp = tp / np.sum(label == 1)

    tp_rate = (tp / (tp + fn)) * 100
    fp_rate = (fp / (fp + tn)) * 100
    tn_rate = (tn / (tn + fp)) * 100
    fn_rate = (fn / (fn + tp)) * 100

    return iou, tp_rate, tn_rate, fp_rate, fn_rate

height, width = 20, 20
x = np.zeros((height, width))
y = np.ones((height, width))

x[:10, :10] = 1

iou, tp_rate, tn_rate, fp_rate, fn_rate = compute_metrics(x, y)
print(iou, tp_rate, tn_rate, fp_rate, fn_rate)
