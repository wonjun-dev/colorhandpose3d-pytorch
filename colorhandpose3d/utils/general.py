import math

import numpy as np
import torch
import torch.nn.functional as F
from roi_align.crop_and_resize import CropAndResizeFunction

import dilation2d


def dilation(x, kernel, stride=[1, 1], rates=[1, 1], padding=[0, 0]):
    """Computes the dilation of a 4D input with a 3D kernel.

    Args:
        x - (batch_size, channel_size, height, width): Input `Tensor`.
        kernel - (chanel_size, height, width): Dilation kernel.
        stride - (stride_height, stride_width): A list of `int`s determining
            the stride of the `kernel`.
        rates - (rate_height, rate_width): A list of `int`s determining the stride
            for atrous morphological dilation.
        padding - (padding_height, padding_width): A list of `int`s defining the amount
            of padding to add to the input `Tensor`.

    Returns:
        A `Tensor` with the same type as `x`.
    """
    # TODO(Alex): Check that the dilation rate and kernel size are appropriate given the input size.
    assert len(x.shape) == 4, "Input must be 4D (N, C, H, W)"
    assert len(kernel.shape) == 3, "Kernel must be 3D (C, H, W)"

    # Calculate output height and width
    output_height = math.floor((x.shape[2] + 2 * padding[0] - kernel.shape[1]) / stride[0]) + 1
    output_width = math.floor((x.shape[3] + 2 * padding[1] - kernel.shape[2]) / stride[1]) + 1

    output = torch.zeros(x.shape[0], x.shape[1], output_height, output_width)

    # C++ implementation
    dilation2d.dilation2d(x, kernel, stride[0], stride[1], rates[0], rates[1], padding[0], padding[1], output)

    return output

def max_coordinate_dense(x):
    """Calculates the x, y coordinates of the maximum value (per channel) in a matrix.

    Args:
        x - (batch_size, channel_size, height, width): Input tensor.

    Returns:
        A tensor of size (batch_size, channel_size, height, width) where each batch item
        is a zero-matrix per channel except for the location of the largest calculated value.
    """

    s = x.shape

    if len(s) == 3:
        output = torch.zeros_like(x, dtype=torch.int32)
        coords = x.view(s[0], -1)
        _, max_coords = torch.max(coords, -1)
        X = torch.remainder(max_coords[:], s[1])
        Y = max_coords[:] / s[2]
        output[:, Y, X] = 1

    return output

def single_obj_scoremap(mask, filter_size=21):
    """Calculates the most likely object given the segmentation score map."""

    padding_size = math.floor(filter_size / 2)
    s = mask.shape
    assert len(s) == 4, "Scoremap must be 4D."

    scoremap_softmax = F.softmax(mask, dim=1)
    scoremap_softmax = scoremap_softmax[:, 1, :, :].unsqueeze(0)
    scoremap_fg_vals, scoremap_fg_idxs = scoremap_softmax.max(dim=1, keepdim=False)
    detmap_fg = torch.round(scoremap_fg_vals)

    max_loc = max_coordinate_dense(scoremap_fg_vals).to(torch.float32)

    objectmap_list = []
    kernel_dil = torch.ones(1, filter_size, filter_size) / float(filter_size * filter_size)

    for i in range(s[0]):
        # create initial object map
        objectmap = max_loc[i]

        num_passes = max(s[2], s[3]) // (filter_size // 2)
        for j in range(num_passes):
            objectmap = torch.reshape(objectmap, [1, 1, s[2], s[3]])
            objectmap_dil = dilation(objectmap, kernel_dil, padding=[padding_size, padding_size])
            objectmap_dil = torch.reshape(objectmap_dil, [s[2], s[3]])
            objectmap = torch.round(detmap_fg[i, :, :] * objectmap_dil)

        objectmap = torch.reshape(objectmap, [1, s[2], s[3]])
        objectmap_list.append(objectmap)

    return torch.stack(objectmap_list)

def calc_center_bb(binary_class_mask):
    """Calculate the bounding box of the object in the binary class mask.

    Args:
        binary_class_mask - (batch_size x H x W): Binary mask isolating the hand.

    Returns:
        centers - (batch_size x 2): Center of mass calculation of the hand.
        bbs - (batch_size x 4): Bounding box of containing the hand. [x_min, y_min, x_max, y_max]
        crops - (batch_size x 2): Size of crop defined by the bounding box.
    """

    binary_class_mask = binary_class_mask.to(torch.int32)
    binary_class_mask = torch.eq(binary_class_mask, 1)
    if len(binary_class_mask.shape) == 4:
        binary_class_mask = binary_class_mask.squeeze(1)

    s = binary_class_mask.shape
    assert len(s) == 3, "binary_class_mask must be 3D."

    bbs = []
    centers = []
    crops = []

    for i in range(s[0]):
        y_min = binary_class_mask[i].nonzero()[:, 0].min()
        x_min = binary_class_mask[i].nonzero()[:, 1].min()
        y_max = binary_class_mask[i].nonzero()[:, 0].max()
        x_max = binary_class_mask[i].nonzero()[:, 1].max()

        start = torch.stack([y_min, x_min])
        end = torch.stack([y_max, x_max])
        bb = torch.stack([start, end], 1)
        bbs.append(bb)

        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center = torch.stack([center_y, center_x])
        centers.append(center)

        crop_size_x = x_max - x_min
        crop_size_y = y_max - y_min
        crop_size = max(crop_size_y, crop_size_x)
        crops.append(crop_size)

    bbs = torch.stack(bbs)
    centers = torch.stack(centers)
    crops = torch.stack(crops)

    return centers, bbs, crops


def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """Crops an image.

    Args:
        image - Tensor (batch_size, C, H, W): Images to be cropped.
        crop_location - Tensor (batch_size, 2): Height and width locations to crop.
        crop_size - int: Size of the crop.
        scale - float: Scale factor.

    Returns:
        image_crop - Tensor (batch_size, C, crop_size, crop_size): Cropped images
    """

    s = image.shape
    assert len(s) == 4, "Image needs to be of shape (B x C x H x W)"
    crop_location = crop_location.to(torch.float32)

    crop_size_scaled = float(crop_size) / scale
    y1 = crop_location[:, 0] - crop_size_scaled // 2
    y2 = y1 + crop_size_scaled
    x1 = crop_location[:, 1] - crop_size_scaled // 2
    x2 = x1 + crop_size_scaled
    y1 /= s[2]
    y2 /= s[2]
    x1 /= s[3]
    x2 /= s[3]
    boxes = torch.stack([y1, x1, y2, x2], -1).to(torch.float32)

    box_ind = torch.arange(0, s[0], dtype=torch.int32)
    image_crops = CropAndResizeFunction(crop_size, crop_size, 0)(image, boxes, box_ind)

    return image_crops


def detect_keypoints(scoremaps):
    """Detect keypoints using the scoremaps provided by PoseNet.

    Args:
        scoremaps - numpy array (num_scoremaps x H x W): Scoremaps of a single
            sample.

    Returns:
        keypoint_coords - numpy array (num_scoremaps x 2): Coordinates of each
            keypoint.
    """

    s = scoremaps.shape
    assert len(s) == 3, "Input must be 3D."

    keypoint_coords = np.zeros((s[0], 2))

    for i in range(s[0]):
        v, u = np.unravel_index(np.argmax(scoremaps[i]), (s[1], s[2]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u

    return keypoint_coords


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """Plots the 2D pose estimates into a matplotlib figure."""
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)