from fileinput import filename
import os
import sys
import argparse
from pprint import pprint
from time import time, sleep
import glob
from turtle import color
from cv2 import FILE_NODE_MAP, destroyWindow
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
import lasso

try:
    detector = hand_tracker.HandTracker(
        2, 0.7, 0.5
    )
except Exception as e:
    logger.error(e)
    exit(1)


_lasso = lasso.load_lasso_model('lasso_model.pkl')

def calculate_lasso_predict_mean(joint_angles :list):
    frame_cnt = 0
    lasso_predict_sum = 0.0

    for joint_angle in joint_angles:
        x = []
        x.extend(list(joint_angle['Left']))
        x.extend(list(joint_angle['Right']))
        pre = _lasso.predict([x])[0]
        pre = round(pre,2) # 小数点2桁へ丸める
        frame_cnt += 1
        lasso_predict_sum += pre
        
    return lasso_predict_sum/frame_cnt



def calculate_landmarks(input_video_path :str):
    """
    ビデオからランドマーク情報を計算して[ビデオ名.pkl]形式で保存する．
    """
    # print(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    ds = detection_state.DetectionState()

    # #動画サイズ取得
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # #フレームレート取得
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # #フォーマット指定
    # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # #注）グレースケールの画像を出力する場合は第5引数に0を与える
    # writer = cv2.VideoWriter('./output.mp4', fmt, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        tmp_image = copy.deepcopy(frame)

        if detector.detect(frame):
            tmp_image, tmp_landmark_dict = detector.draw(tmp_image)
            ds.update_landmarks(tmp_landmark_dict)
        
        cv2.imshow(input_video_path, tmp_image)
        # writer.write(tmp_image)
        key = cv2.waitKey(1) & 0xFF #waitkeyがないとエラーが出る
        if key == ord('q'):
            sys.exit()
            # cv2.destroyWindow(input_video_path)
            # break
    cap.release()
    # writer.release()
    # print("---------------------------")
    cv2.destroyWindow(input_video_path)
    return ds


# 動画を左右反転させる(左利き用):動画を左右反転して保存する用なのでリアルタイム処理ではない．
def flip_left_right(input_video_path :str):

    # video_name = Path(video_path).stem  # 拡張子抜きのファイル名

    cap = cv2.VideoCapture(input_video_path)

    #動画サイズ取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #フレームレート取得
    fps = cap.get(cv2.CAP_PROP_FPS)

    #フォーマット指定
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #注）グレースケールの画像を出力する場合は第5引数に0を与える
    writer = cv2.VideoWriter(f'./output.mp4', fmt, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        tmp_image = copy.deepcopy(frame)

        if detector.detect(frame):
            tmp_image, tmp_landmark_dict = detector.draw(tmp_image)
        
        cv2.imshow(input_video_path, tmp_image)
        frame = cv2.flip(frame,1)
        writer.write(frame)
        key = cv2.waitKey(1) & 0xFF #waitkeyがないとエラーが出る
        if key == ord('q'):
            sys.exit()
            # cv2.destroyWindow(input_video_path)
            # break
    cap.release()
    writer.release()
    print("---------------------------")
    cv2.destroyWindow(input_video_path)



def calculate_joint_angle(landmarks :list):
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
    for landmark in landmarks:
        joint_angle = {'Left':[], 'Right':[]}
        
        # 手を1つしか検出していないときはスルーする
        if landmark['Left']==[] or landmark['Right']==[]:
            continue

        for hand in ['Left', 'Right']:
            angles = []
            
            # 関節のつながりからなす角度を計算する
            for connection in connections:
                A = np.array(landmark[hand][connection[0]])
                B = np.array(landmark[hand][connection[1]])
                C = np.array(landmark[hand][connection[2]])
                BA = A-B
                BC = C-B
                norm_AB = np.linalg.norm(BA)
                norm_BC = np.linalg.norm(BC)
                inner_AB_BC = np.inner(BA, BC)
                angle_rad = np.arccos(inner_AB_BC/(norm_AB*norm_BC))
                angle_deg = math.degrees(angle_rad)
                angles.append(angle_deg)
            joint_angle[hand] = angles
        joint_angles.append(joint_angle)
    return joint_angles


def calculate_variance(joint_angles :list) -> dict:
    """
    時系列データにおける分散を計算する
    """
    # 内包表記でスマートに！
    left  = np.array([joint_angle['Left'] for joint_angle in joint_angles])
    right = np.array([joint_angle['Right'] for joint_angle in joint_angles])
    return {'Left' :np.var(left, axis=0), 'Right' :np.var(right, axis=0)}


def calculate_mean(joint_angles :list) -> dict:
    """
    時系列データにおける平均を計算する
    """
    left  = np.array([joint_angle['Left'] for joint_angle in joint_angles])
    right = np.array([joint_angle['Right'] for joint_angle in joint_angles])
    return {'Left' :np.mean(left, axis=0), 'Right' :np.mean(right, axis=0)}


def calculate_operation_time(input_video_path :str) -> int:
    """
    作業秒数（動画の時間）を計算する
    """
    cap = cv2.VideoCapture(input_video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return int(frame_count/fps)


def apply_moving_average(landmarks :list) -> list:
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
                # ds = detection_state.load_detection_state(pkl_path=output_pkl_path)
                # continue
                pass
            
            ds = calculate_landmarks(input_video_path=input_video_path)
            
            ds.landmarks = apply_moving_average(ds.landmarks)   # 移動平均を適用する
            
            ds.operation_time = calculate_operation_time(input_video_path=input_video_path) # 動画の時間を計算する
            ds.joint_angles = calculate_joint_angle(ds.landmarks)                           # 関節の角度を計算する
            if ds.joint_angles == []:
                print(f'手を検出しなかった::{video_path}')
                continue
            ds.joint_angle_mean = calculate_mean(ds.joint_angles)                           # 関節の角度の平均を計算する
            ds.joint_angle_var = calculate_variance(ds.joint_angles)                        # 関節の角度の分散を計算する
            detection_state.save_detection_state(ds=ds, output_pkl_path=output_pkl_path)    # ランドマークを保存する

            # グラフの表示
            # graph = Graphic_3D(ds.landmarks)
            # graph.plot(animation_path=f'{BASE_DIR_PATH}/{dir_name}/{video_name}.gif')


def execute_calculation(input_video_path :str) -> None:
    ds = detection_state.DetectionState()






# if __name__ == "__main__":
#     main()
    
# for i, l in enumerate(_list):
#     ax1.scatter(0 if l[3]==0 else 1 ,l[0], color='red' if l[3]==0 else 'blue')
#     ax2.scatter(0 if l[3]==0 else 1 ,l[1], color='red' if l[3]==0 else 'blue')
#     ax3.scatter(0 if l[3]==0 else 1 ,l[2], color='red' if l[3]==0 else 'blue')

# plt.plot()
# plt.show()


# ds = detection_state.load_detection_state("C:\\Users\\shino\\Desktop\\R4\\GraduationResearch\\Program\\data\\4\\5.pkl")
# print(str(ds))

# ds = calculate_joint_angle(ds.landmarks)
# print(ds)
# 手の検出ができていないところがあるので取り除こう

# ds = calculate_landmarks("C:\\Users\\shino\\Desktop\\R4\\GraduationResearch\\Program\\data\\4\\2.mp4")            
# ds.landmarks = apply_moving_average(ds.landmarks)   # 移動平均を適用する
# joint_angles = calculate_joint_angle(ds.landmarks)                           # 関節の角度を計算する
# if joint_angles == []:
#     print("A")

# ds = calculate_landmarks("C:\\Users\\shino\\Desktop\\R4\\GraduationResearch\\Program\\data\\4\\5.mp4")            
# ds.landmarks = apply_moving_average(ds.landmarks)   # 移動平均を適用する
# joint_angles = calculate_joint_angle(ds.landmarks)                           # 関節の角度を計算する
# if joint_angles == []:
#     print("A")