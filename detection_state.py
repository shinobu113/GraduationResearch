import numpy as np
import cv2
import pickle
import pprint

import hand_tracker

class DetectionState(object):
    # インスタンス変数とクラス変数の違いに注意する．(前者を使用しないとpklで上手くシリアライズ化できない)    
    def __init__(self) -> None:
        self.landmarks = []             # ランドマークの時系列データ
        self.latest_landmark_dict = {}  # 最新のランドマークを保持．
        self.joint_angles = []           # 関節の角度
        self.gender = 'man'             # 性別
        self.dominant_hand = 'Right'    # 利き手('Right' or 'Left')←検出した手(handness)とは異なることに注意する．
        self.operation_time = 0         # はんだ付けの作業時間
        self.joint_angle_mean = {}
        self.joint_angle_var = {}

    def update_landmarks(self, landmark_dict :dict) -> None:
        self.landmarks.append(landmark_dict)
        self.latest_landmark_dict = landmark_dict
    
    # クラス変数を表示するための関数
    def __str__(self):
        return f'landmarks : {self.landmarks}\n \
                joint_angles : {self.joint_angles}\n \
                gender : {self.gender}\n \
                dominant_hand : {self.dominant_hand}\n \
                operation_time : {self.operation_time}\n \
                joint_angle_mean : {self.joint_angle_mean}\n \
                joint_angle_var : {self.joint_angle_var}'


# pkl形式で保存する
def save_detection_state(ds :DetectionState, output_pkl_path :str):
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(ds, f)



# 保存したpklを読み込む
def load_detection_state(pkl_path :str):
    with open(pkl_path, 'rb') as f:
        ds = pickle.load(f)
    return ds
