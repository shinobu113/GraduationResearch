import os
import argparse
from pprint import pprint
from time import time, sleep
import glob
from cv2 import FILE_NODE_MAP
import pickle
# import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
from loguru import logger
from graphic import Graphic_3D

import hand_tracker
import detection_state

try:
    detector = hand_tracker.HandTracker(
        2, 0.7, 0.5
    )
except Exception as e:
    logger.error(e)
    exit(1)


def calculate_landmarks(video_path :str):
    """
    ビデオからランドマーク情報を計算してpkl形式で保存する．
    """
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    ds = detection_state.DetectionState()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        tmp_image = copy.deepcopy(frame)

        if detector.detect(frame):
            tmp_image, tmp_landmark_dict = detector.draw(tmp_image)
            ds.update_landmarks(tmp_landmark_dict)
        
        # cv2.imshow("A", tmp_image)
        key = cv2.waitKey(1) & 0xFF #waitkeyがないとエラーが出る
        if key == ord('q'):
            break
    print("---------------------------")
    return ds


def calculate_joint_angle(ds :detection_state.DetectionState):
    """
    ランドマーク情報から関節の開き具合(角度)を計算する
    """
    pass

def apply_moving_average(landmarks :list):
    """
    ランドマークの情報に移動平均を適用して平滑化する．
    """
    point_num = min(5, len(landmarks)) # 移動平均点数
    res_landmarks = []
    for i in range(point_num-1, len(landmarks)):
        land_left  = []
        land_right = []
        for j in range(point_num):
            if landmarks[i-j]['Left'] != []:
                land_left.append(landmarks[i-j]['Left'])
            if landmarks[i-j]['Right'] != []:
                land_right.append(landmarks[i-j]['Right'])
        # print(land_left, land_right)
        land_left  = np.array(land_left)
        land_right = np.array(land_right)
        
        if land_left.size==0: land_left_mean=[]
        else                : land_left_mean=np.mean(land_left, axis=0)

        if land_right.size==0: land_right_mean=[]
        else                 : land_right_mean=np.mean(land_right, axis=0)

        res_landmarks.append({'Left':list(land_left_mean), 'Right':list(land_right_mean)})
    
    return res_landmarks


            



def main():
    
    # data内のフォルダを取得し、その中の動画に対して解析を行う
    FOLDER_PATH = './data/'
    folders = os.listdir(FOLDER_PATH)
    
    for folder in folders:
        video_path = glob.glob(f'{FOLDER_PATH}/{folder}/*.mp4')
        
        # landmarks.pklが既にあれば読み込むだけでよい
        if os.path.isfile(f'{FOLDER_PATH}/{folder}/landmarks.pkl'):
            ds = detection_state.load_detection_state(path=folder)
        else:
            ds = calculate_landmarks(video_path[0]) # ビデオが2つ以上ある場合は修正が必要．
            detection_state.save_detection_state(ds=ds, path=folder)
        # print(ds.landmarks)
        graph = Graphic_3D(ds.landmarks)
        graph.plot(path=f'./data/{folder}')
        # ds.landmarks = apply_moving_average(ds.landmarks)


if __name__ == "__main__":
    main()