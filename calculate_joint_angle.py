from fileinput import filename
import os
import sys
import argparse
from pprint import pprint
from time import time, sleep
import glob
from cv2 import FILE_NODE_MAP
import pickle
import math
from pathlib import Path

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


def calculate_landmarks(input_video_path :str):
    """
    ビデオからランドマーク情報を計算して[ビデオ名.pkl]形式で保存する．
    """
    print(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    ds = detection_state.DetectionState()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        tmp_image = copy.deepcopy(frame)

        if detector.detect(frame):
            tmp_image, tmp_landmark_dict = detector.draw(tmp_image)
            ds.update_landmarks(tmp_landmark_dict)
        
        cv2.imshow(input_video_path, tmp_image)
        key = cv2.waitKey(1) & 0xFF #waitkeyがないとエラーが出る
        if key == ord('q'):
            sys.exit()
    print("---------------------------")
    return ds



def calculate_joint_angle(ds :detection_state.DetectionState):
    """
    ランドマーク情報から関節の開き具合(角度)を計算する
    """
    connections = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [0, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [9, 10, 11],
        [10, 11, 12],
        [13, 14, 15],
        [14, 15, 16],
        [17, 18, 19],
        [18, 19, 20]
    ]
    
    joint_angles = []
    for landmark in ds.landmarks:
        joint_angle = {'Left':[], 'Right':[]}
        
        for hand in ['Left', 'Right']:
            angles = []
            if landmark[hand] == []:
                joint_angle[hand] = [10000]
                # !欠損値として処理する？
                continue
            
            # 関節のつながりからなす角度を計算する
            for connection in connections:
                A = landmark[hand][connection[0]]
                B = landmark[hand][connection[1]]
                C = landmark[hand][connection[2]]
                AB = B-A
                BC = C-B
                norm_AB = np.linalg.norm(AB)
                norm_BC = np.linalg.norm(BC)
                inner_AB_BC = np.inner(AB, BC)
                angle_rad = np.arccos(inner_AB_BC/(norm_AB*norm_BC))
                angle_deg = math.degrees(angle_rad)
                angles.append(angle_deg)
                # print(angle_deg)
            joint_angle[hand] = angles
        joint_angles.append(joint_angle)
    print(joint_angles)        



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
        land_left  = np.array(land_left)
        land_right = np.array(land_right)
        
        if land_left.size==0: land_left_mean=[]
        else                : land_left_mean=np.mean(land_left, axis=0)

        if land_right.size==0: land_right_mean=[]
        else                 : land_right_mean=np.mean(land_right, axis=0)

        res_landmarks.append({'Left':list(land_left_mean), 'Right':list(land_right_mean)})
    
    return res_landmarks



def main():
    """
    data内のフォルダを取得し、その中の動画に対して解析を行う
    """
    BASE_DIR_PATH = './data/'
    dir_names = os.listdir(BASE_DIR_PATH)
    for dir_name in dir_names:
        video_paths = glob.glob(f'{BASE_DIR_PATH}/{dir_name}/*.mp4')
        
        for video_path in video_paths:
            video_name = Path(video_path).stem  # 拡張子抜きのファイル名
            
            if video_name=='original':  # 動画の元データには解析を行わない
                continue

            input_video_path = video_path
            output_pkl_path  = f'{BASE_DIR_PATH}/{dir_name}/{video_name}.pkl'
            
            if os.path.isfile(output_pkl_path):
                ds = detection_state.load_detection_state(pkl_path=output_pkl_path)
            else:
                ds = calculate_landmarks(input_video_path=input_video_path)
                detection_state.save_detection_state(ds=ds, output_pkl_path=output_pkl_path)
            
            # 移動平均を適用する
            ds.landmarks = apply_moving_average(ds.landmarks)
            calculate_joint_angle(ds)
            
            # グラフの表示
            # graph = Graphic_3D(ds.landmarks)
            # graph.plot(animation_path=f'{BASE_DIR_PATH}/{dir_name}/{video_name}.gif')


if __name__ == "__main__":
    main()