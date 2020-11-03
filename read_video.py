# 저장 된 비디오를 읽고 3d pose estimation 이 된 frame을 다시 비디오로 저장.

import os
import sys

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision

model_path = os.path.abspath(os.path.join(".."))
if model_path not in sys.path:
    sys.path.append(model_path)

from colorhandpose3d.model.ColorHandPose3D import ColorHandPose3D
from colorhandpose3d.utils.general import *
from colorhandpose3d.utils.transforms import *

from matplotlib import pyplot as plt
from tqdm import tqdm
import time

# Initialize models and load weights
weight_path = "../saved/"
use_cuda = torch.cuda.is_available()
use_cuda = False

chp3d = ColorHandPose3D(weight_path)
if use_cuda:
    chp3d.cuda()


# Initialize OpenCV object
cap = cv2.VideoCapture("../outputs/before_test.avi")  # Open video
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(frame_width, frame_height, length, end=",")

# Define image transforms
transform0 = torchvision.transforms.ToPILImage()
transform1 = torchvision.transforms.ToTensor()
transform2 = torchvision.transforms.Resize(256)


def main():

    # Preprocessing recorded frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()  # frame: numpy array, (H480, W640, C3)

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            sample_original = transform1(frame).unsqueeze(0)  # (C3, H480, W640)
            sample = sample_original - 0.5
            hand_side = torch.tensor([[1.0, 0.0]])

            if use_cuda:
                sample = sample.cuda()
                hand_side = hand_side.cuda()

            frames.append([sample, sample_original, hand_side])

        else:
            break

    # Pass frames through chp3d
    passFrames(frames)

    cap.release()
    cv2.destroyAllWindows()


def passFrames(frames):

    # Video writer
    # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    # out = cv2.VideoWriter("../outputs/after_test.avi", fourcc, 25.0, (640, 480))

    for i, frame in tqdm(enumerate(frames)):
        sample = frame[0]
        sample_original = frame[1]
        hand_side = frame[2]

        # Pass frame through chp3d
        s = time.time()
        coords_xyz_rel_normed, keypoint_scoremap, image_crop, centers, scale_crop = chp3d(
            sample, hand_side
        )
        print("Time for processing 1 frame: {}s".format(time.time() - s))

        # Back to CPU
        if use_cuda is True:
            coords_xyz_rel_normed = coords_xyz_rel_normed.cpu()
            keypoint_scoremap = keypoint_scoremap.cpu()
            image_crop = image_crop.cpu()
            centers = centers.cpu()
            scale_crop = scale_crop.cpu()

        keypoint_coords3d = coords_xyz_rel_normed.detach().numpy()
        keypoint_coords3d = keypoint_coords3d.squeeze()

        keypoint_coords_crop = detect_keypoints(keypoint_scoremap[0].detach().numpy())
        keypoint_coords = transform_cropped_coords(keypoint_coords_crop, centers, scale_crop, 256)

        img = transform0(sample_original.squeeze())
        fig = plt.figure()
        ax1 = fig.add_subplot()
        plt.imshow(img)
        plot_hand(keypoint_coords, ax1)
        plt.axis("off")
        plt.savefig("..after_frames/outputs/{}.png".format(i))
        plt.close()


if __name__ == "__main__":
    main()
