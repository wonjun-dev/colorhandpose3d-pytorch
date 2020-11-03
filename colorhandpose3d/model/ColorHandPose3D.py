import os

import torch

from .HandSegNet import HandSegNet
from .PoseNet import PoseNet
from .PosePrior import PosePrior
from .ViewPoint import ViewPoint
from .HandNet import HandNet
from ..utils.general import *
from ..utils.transforms import *
import time


class ColorHandPose3D(torch.nn.Module):
    """ColorHandPose3D predicts the 3D joint location of a hand given the
    cropped color image of a hand."""

    def __init__(self, weight_path=None, crop_size=None, num_keypoints=None):
        super(ColorHandPose3D, self).__init__()
        self.handsegnet = HandSegNet()
        self.handnet = HandNet()
        self.posenet = PoseNet()
        self.poseprior = PosePrior()
        self.viewpoint = ViewPoint()

        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        if num_keypoints is None:
            self.num_keypoints = 21
        else:
            self.num_keypoints = num_keypoints

        # Load weights
        if weight_path is not None:
            self.handsegnet.load_state_dict(
                torch.load(os.path.join(weight_path, "handsegnet.pth.tar"))
            )
            self.handnet.load_state_dict(torch.load(os.path.join(weight_path, "handnet.pth.tar")))
            self.posenet.load_state_dict(torch.load(os.path.join(weight_path, "posenet.pth.tar")))
            self.poseprior.load_state_dict(
                torch.load(os.path.join(weight_path, "poseprior.pth.tar"))
            )
            self.viewpoint.load_state_dict(
                torch.load(os.path.join(weight_path, "viewpoint.pth.tar"))
            )

    def forward(self, x, hand_sides, deploy=False):
        """Forward pass through the network.

        Args:
            x - Tensor (B x C x H x W): Batch of images.
            hand_sides - Tensor (B x 2): One-hot vector indicating if the hand
                is left or right.

        Returns:
            coords_xyz_rel_normed (B x N_k x 3): Normalized 3D coordinates of
                the joints, where N_k is the number of keypoints.
        """

        if deploy:
            x1 = x[0]
            x2 = x[1]
            # detect existence of hand
            s = time.time()
            hand_prob = self.handnet(x1)
            print(hand_prob)
            print("Handnet forward time: {}".format(time.time() - s))

            if hand_prob.item() >= 0.0:
                # Segment the hand
                s = time.time()
                hand_scoremap = self.handsegnet.forward(x2)
                print("Handsegnet forward time: {}".format(time.time() - s))

                # Calculate single highest scoring object
                s = time.time()
                hand_mask = single_obj_scoremap(hand_scoremap, self.num_keypoints)
                print("hand mask sum: {}".format(torch.sum(hand_mask)))
                print("Calculate single highest scoring object time: {}".format(time.time() - s))

                # crop and resize
                s = time.time()
                centers, _, crops = calc_center_bb(hand_mask)
                crops = crops.to(torch.float32)

                crops *= 1.25
                scale_crop = torch.min(
                    torch.max(self.crop_size / crops, torch.tensor(0.25, device=x2.device)),
                    torch.tensor(5.0, device=x2.device),
                )
                image_crop = crop_image_from_xy(x2, centers, self.crop_size, scale_crop)
                print("Crop and resize time: {}".format(time.time() - s))

                # detect 2d keypoints
                s = time.time()
                keypoints_scoremap = self.posenet(image_crop)
                print("Posenet forward time: {}".format(time.time() - s))

                # estimate 3d pose
                s = time.time()
                coord_can = self.poseprior(keypoints_scoremap, hand_sides)
                print("Posepriornet forward time: {}".format(time.time() - s))

                s = time.time()
                rot_params = self.viewpoint(keypoints_scoremap, hand_sides)
                print("Viewpoint forward time: {}".format(time.time() - s))

                # get normalized 3d coordinates
                s = time.time()
                rot_matrix = get_rotation_matrix(rot_params)
                cond_right = torch.eq(torch.argmax(hand_sides, 1), 1)
                cond_right_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(
                    1, self.num_keypoints, 3
                )
                coords_xyz_can_flip = flip_right_hand(coord_can, cond_right_all)
                coords_xyz_rel_normed = coords_xyz_can_flip @ rot_matrix
                print("Get normalized 3d coordinates time: {}".format(time.time() - s))

                # flip left handed inputs wrt to the x-axis for Libhand compatibility.
                s = time.time()
                coords_xyz_rel_normed = flip_left_hand(coords_xyz_rel_normed, cond_right_all)
                print("Flip left handed input time: {}".format(time.time() - s))

                # scale heatmaps
                s = time.time()
                keypoints_scoremap = F.interpolate(
                    keypoints_scoremap, self.crop_size, mode="bilinear", align_corners=False
                )
                print("ScaLe heatmaps time {}".format(time.time() - s))

                return (
                    coords_xyz_rel_normed,
                    keypoints_scoremap,
                    image_crop,
                    centers,
                    scale_crop,
                    hand_mask,
                )
            else:
                print("no hand")
                return None, None, None, None, None, None

        else:
            x2 = x[-1]
            # Segment the hand
            s = time.time()
            hand_scoremap = self.handsegnet.forward(x2)
            print("Handsegnet forward time: {}".format(time.time() - s))

            # Calculate single highest scoring object
            s = time.time()
            hand_mask = single_obj_scoremap(hand_scoremap, self.num_keypoints)
            print("hand mask sum: {}".format(torch.sum(hand_mask)))
            print("Calculate single highest scoring object time: {}".format(time.time() - s))

            # crop and resize
            s = time.time()
            centers, _, crops = calc_center_bb(hand_mask)
            crops = crops.to(torch.float32)

            crops *= 1.25
            scale_crop = torch.min(
                torch.max(self.crop_size / crops, torch.tensor(0.25, device=x2.device)),
                torch.tensor(5.0, device=x2.device),
            )
            image_crop = crop_image_from_xy(x2, centers, self.crop_size, scale_crop)
            print("Crop and resize time: {}".format(time.time() - s))

            # detect 2d keypoints
            s = time.time()
            keypoints_scoremap = self.posenet(image_crop)
            print("Posenet forward time: {}".format(time.time() - s))

            # estimate 3d pose
            s = time.time()
            coord_can = self.poseprior(keypoints_scoremap, hand_sides)
            print("Posepriornet forward time: {}".format(time.time() - s))

            s = time.time()
            rot_params = self.viewpoint(keypoints_scoremap, hand_sides)
            print("Viewpoint forward time: {}".format(time.time() - s))

            # get normalized 3d coordinates
            s = time.time()
            rot_matrix = get_rotation_matrix(rot_params)
            cond_right = torch.eq(torch.argmax(hand_sides, 1), 1)
            cond_right_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(1, self.num_keypoints, 3)
            coords_xyz_can_flip = flip_right_hand(coord_can, cond_right_all)
            coords_xyz_rel_normed = coords_xyz_can_flip @ rot_matrix
            print("Get normalized 3d coordinates time: {}".format(time.time() - s))

            # flip left handed inputs wrt to the x-axis for Libhand compatibility.
            s = time.time()
            coords_xyz_rel_normed = flip_left_hand(coords_xyz_rel_normed, cond_right_all)
            print("Flip left handed input time: {}".format(time.time() - s))

            # scale heatmaps
            s = time.time()
            keypoints_scoremap = F.interpolate(
                keypoints_scoremap, self.crop_size, mode="bilinear", align_corners=False
            )
            print("ScaLe heatmaps time {}".format(time.time() - s))

            return (
                coords_xyz_rel_normed,
                keypoints_scoremap,
                image_crop,
                centers,
                scale_crop,
                hand_mask,
            )
