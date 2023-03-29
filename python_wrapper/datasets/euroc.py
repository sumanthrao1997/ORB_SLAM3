#!/usr/bin/env python3
import cv2
import numpy as np
import os


class EUROCDataset:
    def __init__(self, data_source):

        self.data_source = os.path.abspath(data_source)
        self.cam0 = np.loadtxt(
            fname=f"{self.data_source}/mav0/cam0/data.csv",
            delimiter=",",
            dtype=str,
        )
        self.cam1 = np.loadtxt(
            fname=f"{self.data_source}/mav0/cam1/data.csv",
            delimiter=",",
            dtype=str,
        )
        self.cam_timestamps = self.cam0[:, 0].astype(np.float64)

    def __getitem__(self, idx):
        imL_name = self.cam0[idx, 1]
        imR_name = self.cam1[idx, 1]
        tframe = self.cam_timestamps[idx]
        imL = cv2.imread(
            f"{self.data_source}/mav0/cam0/data/{imL_name}",
            cv2.IMREAD_UNCHANGED,
        )
        imR = cv2.imread(
            f"{self.data_source}/mav0/cam1/data/{imR_name}",
            cv2.IMREAD_UNCHANGED,
        )
        return imL, imR, tframe

    def __len__(self):
        return len(self.cam)
