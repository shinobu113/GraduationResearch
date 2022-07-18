import os
import csv
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import mpl_toolkits.mplot3d.art3d as art3d
from loguru import logger

pkl_name = 'landmarks.pkl'
csv_name = 'landmarks.csv'


# 環境変数の設定
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"



def write_datas(datas: list, file_name: str) -> None:
    """ランドマークをpickle, csvで保存
        Args:
            list: ランドマーク
    """
    df = pd.DataFrame(datas)
    df.to_csv(f'{file_name}.csv', index=False)

    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(datas, f)


def show_landmarks(pkl_mode: bool, landmark: np.ndarray = np.array([])) -> None:
    """ランドマークの時間的変化を表示する
        Args:
            bool: csvを読み込んで表示するか
            list: ランドマーク
    """
    if pkl_mode:
        with open(pkl_name, 'rb') as f:
            landmark = list(pickle.load(f))
    landmarks = np.array(landmark)

    # landmarksのデータ加工
    # landmarks = fit_landmarks_to_standard(landmarks)
    landmarks = moving_average(landmarks)

    try:
        for i in range(5):
            plt.plot(landmarks[:, i, 0], label=str(i))
        plt.legend()
        plt.show()
    except Exception as e:
        logger.error(e)
        pass

    show_3d_models(landmarks)


def show_3d_models(landmarks: np.ndarray) -> None:
    """ランドマークを3D座標で表示する
        Args:
            np.ndarray: ランドマーク
    """
    if landmarks.size == 0:
        print("No landmarks")
        return

    landmarks[:, :, 0], landmarks[:, :, 1] = np.array(
        landmarks[:, :, 1]), np.array(landmarks[:, :, 0])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=35, azim=20)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    connections = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [5, 9, 13, 17, 0, 5]
    ]
    for conn in connections:
        plot_xyz = []
        for land in conn:
            plot_xyz.append(landmarks[0, land])
        plot_xyz = np.array(plot_xyz)
        line = art3d.Line3D(
            plot_xyz[:, 0], plot_xyz[:, 1], plot_xyz[:, 2], color='c', linewidth=6)
        ax.add_line(line)

    ax.plot(landmarks[0, :, 0], landmarks[0, :, 1], landmarks[0,
            :, 2], "o", color="#457b9d", ms=8, linestyle='None')
    for i, label in enumerate(landmarks[0]):
        ax.text(label[0], label[1], label[2], s=str(i))

    scale = 1.2
    ax.set_xlim3d(min(landmarks[0, :, 0])/scale, max(landmarks[0, :, 0])*scale)
    ax.set_ylim3d(min(landmarks[0, :, 1])/scale, max(landmarks[0, :, 1])*scale)
    ax.set_zlim3d(min(0, *landmarks[0, :, 2])
                  * scale, max(landmarks[0, :, 2])*scale)

    ax.invert_zaxis()

    animation = FuncAnimation(
        fig,
        func=(lambda i: ax.view_init(elev=45, azim=i*5)),
        frames=72,
        interval=100
    )
    #save_datas(animation, 'Animation', 'images//gifs', 'animation', '.gif')
    # dd = cv2.imread('images//acc_images/0.png')
    # save_datas(dd, 'Picture', 'images//caps', 'pic', '.png')
    plt.show()


def save_datas(data: object, data_type: str, save_dir: str, common_name: str, file_extension: str) -> None:
    """アニメーション，画像を目的フォルダに連番で保存
        Args:
            object          : データオブジェクト
            str             : データの種類(Animation or Picture)
            save_dir        : 保存するディレクトリ
            common_name     : ファイルの共通部分の名前
            file_extention  : 拡張子
    """
    if not data_type in ['Animation', 'Picture']:
        print("Wrong data_type")
        return

    cnt = 0
    while(1):
        if not os.path.isfile(f'{os.getcwd()}//{save_dir}//{common_name}{cnt}.{file_extension}'):
            if data_type == 'Animation':
                data.save(
                    f'{save_dir}//{common_name}{cnt}.{file_extension}', writer='imagemagick')
                break
            elif data_type == 'Picture':
                cv2.imwrite(
                    f'{save_dir}//{common_name}{cnt}.{file_extension}', data)
                break
        cnt += 1


def fit_landmarks_to_standard(landmarks: np.ndarray) -> np.ndarray:
    """手首をランドマークの基準に合わせる
        args:
            np.ndarray: ランドマーク
        return:
            np.ndarray: 修正後のランドマーク
    """
    for i in range(len(landmarks)):
        landmarks[i] -= landmarks[i][0]

    return landmarks


def moving_average(landmarks: np.ndarray) -> np.ndarray:
    """ランドマークに移動平均を適用する
        args:
            np.ndarray: ランドマーク
        return:
            np.ndadday: 移動平均後のランドマーク
    """
    average_num = min(25, len(landmarks))
    res_landmarks = []

    for i in range(average_num, len(landmarks)):
        res_landmarks.append(np.mean(landmarks[i-average_num:i, :], axis=0))

    res_landmarks = np.array(res_landmarks)
    return res_landmarks
