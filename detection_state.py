from json import load
import numpy as np
import cv2
import pickle
import pprint
import os
import glob
from pathlib import Path
from attr import field

import hand_tracker
import graphic
class DetectionState(object):
    # インスタンス変数とクラス変数の違いに注意する．(前者を使用しないとpklで上手くシリアライズ化できない)    
    def __init__(self) -> None:
        self.landmarks = []             # ランドマークの時系列データ
        self.latest_landmark_dict = {}  # 最新のランドマークを保持．
        self.joint_angles = []          # 関節の角度
        self.gender = 'man'             # 性別
        self.dominant_hand = 'Right'    # 利き手('Right' or 'Left')←検出した手(handness)とは異なることに注意する．
        self.operation_time = 0         # はんだ付けの作業時間
        self.joint_angle_mean = {}
        self.joint_angle_var = {}
        self.label = 0                  # はんだが良いか悪いかのラベル
        self.lasso_mean_value = 0.0     # Lassoの平均値(ファイル解析時に保存する)

    def update_landmarks(self, landmark_dict :dict) -> None:
        self.landmarks.append(landmark_dict)
        self.latest_landmark_dict = landmark_dict
    
    # クラス変数を表示するための関数
    def __str__(self):
        # return f'landmarks : {self.landmarks}\n \
        #         joint_angles : {self.joint_angles}\n \
        #         gender : {self.gender}\n \
        #         dominant_hand : {self.dominant_hand}\n \
        #         operation_time : {self.operation_time}\n \
        #         joint_angle_mean : {self.joint_angle_mean}\n \
        #         joint_angle_var : {self.joint_angle_var}\n \
        #         label : {self.label}'
        return f'gender : {self.gender}\ndominant_hand : {self.dominant_hand}\noperation_time : {self.operation_time}\njoint_angle_mean : {self.joint_angle_mean}\njoint_angle_var : {self.joint_angle_var}\nLasso_mean_value : {self.lasso_mean_value}\nlabel : {self.label}'


# pkl形式で保存する
def save_detection_state(ds :DetectionState, output_pkl_path :str):
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(ds, f)



# 保存したpklを読み込む
def load_detection_state(pkl_path :str):
    with open(pkl_path, 'rb') as f:
        ds = pickle.load(f)
    return ds


def set_labels():
    labels = [
        [1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1],
        [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1],
        [1,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0],
    ]
    file_pathes = get_file_path_list()
    # print(file_pathes)
    for file_path in file_pathes:
       _split = file_path.split('\\')
       folder_name = _split[-2]
       file_name = _split[-1].split('.')[0]
       ds = load_detection_state(file_path)
       ds.label = labels[int(folder_name)-1][int(file_name)-1]
    
       output_pkl_path  = f'./data/{folder_name}/{file_name}.pkl'
       save_detection_state(ds, output_pkl_path)



def get_file_path_list() -> list:
    """
    解析するファイル(pkl)のパスをリストで取得
    """
    res_file_list = []
    BASE_DIR_PATH = './data'
    dir_names = os.listdir(BASE_DIR_PATH)
    for dir_name in dir_names:
        video_pathes = glob.glob(f'{BASE_DIR_PATH}\\{dir_name}\\*.pkl')
        for video_path in video_pathes:
            video_name = Path(video_path).stem  # 拡張子抜きのファイル名
            if video_name=='original':  # 動画の元データには解析を行わない
                continue
            res_file_list.append(video_path)
    # print(res_file_list)
    return res_file_list


labels = [
        [1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1],
        [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1],
        [1,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        [0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0],
    ]

# files = get_file_path_list()
# for file in files:
#     ds = load_detection_state(file)
#     video_name = Path(file).stem
#     _split = file.split('\\')
#     folder_name = _split[-2]
#     print(f'{file}: {labels[int(folder_name)-1][int(video_name)-1]}  {ds.label}')
#     ds.label = labels[int(folder_name)-1][int(video_name)-1]
#     if int(folder_name) == 7:
#         ds.dominant_hand = "Left"
#     save_detection_state(ds, f'./data/{folder_name}/{video_name}.pkl')

# files = get_file_path_list()
# for file in files:
#     ds = load_detection_state(file)
#     if ds.label==0:
#         print(file)