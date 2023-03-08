#!/usr/bin/env python3
import cv2
import numpy as np
# from build.python_wrapper import orb_slam_pybind as orb
# import open3d as o3d
from pyquaternion import Quaternion
import sys

# sys.path.append("..")
# from utils.cache import get_cache, memoize


class TUMDataset:
    def __init__(self, data_source):
        # Cache
        # self.use_cache = True
        # self.cache = get_cache(directory="cache/tum/")

        self.data_source = data_source

        self.depth_frames = np.loadtxt(fname=f"{self.data_source}/depth.txt", dtype=str)
        self.rgb_frames = np.loadtxt(fname=f"{self.data_source}/rgb.txt", dtype=str)
        self.image_scale = 10;
        self.matches = np.array(
            self.associate(
                self.rgb_frames[:, 0].astype(np.float64).tolist(),
                self.depth_frames[:, 0].astype(np.float64).tolist(),
            )
        )
        gt_list = np.loadtxt(fname=self.data_source + "/" + "groundtruth.txt", dtype=str)
        self.gt_poses = self.load_poses(gt_list)

# ******************* look at evaluation later ********************** 
    # @memoize()
    def load_poses(self, gt_list):
        indices = np.abs(
            (
                np.subtract.outer(
                    gt_list[:, 0].astype(np.float64),
                    self.depth_frames[:, 0].astype(np.float64),
                )
            )
        ).argmin(0)

        xyz = gt_list[indices][:, 1:4]
        rotations = np.array(
            [
                Quaternion(x=x, y=y, z=z, w=w).rotation_matrix
                for x, y, z, w in gt_list[indices][:, 4:]
            ]
        )
        num_poses = rotations.shape[0]
        poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(num_poses, axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = xyz
        return poses

    # @memoize()
    def associate(self, first_list, second_list, offset=0.0, max_difference=0.2):
        matches = []
        for a in first_list:
            temp_matches = [
                (abs(a - (b + offset)), a, b)
                for b in second_list
                if abs(a - (b + offset)) < max_difference
            ]
            if len(temp_matches) != 0:
                diff, first, second = temp_matches[np.argmin(np.array(temp_matches)[:, 0])]
                matches.append((first, second))
        matches.sort()
        return matches

    # @memoize()
    def __getitem__(self, idx):
        rgb_id, depth_id = self.matches[idx]
        # pose = self.gt_poses[idx]
        rgb = cv2.imread(f"{self.data_source}/rgb/{rgb_id:.6f}.png")
        depth = cv2.imread(f"{self.data_source}/depth/{depth_id:.6f}.png")

        return rgb, depth, float(rgb_id)

    def __len__(self):
        return len(self.matches)
