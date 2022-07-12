# -*- coding:utf-8 -*-
from os.path import exists
import argparse
from pprint import pprint
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
from loguru import logger

import hand_tracker
import get_datas
import detection_state

def get_args() -> argparse.Namespace:
    """引数取得

    Returns:
        argparse.Namespace: 取得結果
    """
    parser = argparse.ArgumentParser()

    # setting camera device
    parser.add_argument(
        "--device", help='デバイスカメラ:0  Webカメラ:1', type=int, default=1)
    parser.add_argument("--width", help='capture width', type=int, default=640)
    parser.add_argument(
        "--height", help='capture height', type=int, default=480
    )
    parser.add_argument(
        '--datatype', type=str, default='', help='保存データの形式'
    )
    parser.add_argument(
        '--still_image', type=str, default='', help='静止画のパス'
    )
    parser.add_argument(
        '--overwrite', type=int, default=0, help='ランドマークの上書き'
    )
    parser.add_argument(
        '--max_num_hands', type=int, default=2, help='最大検出手数'
    )
    parser.add_argument(
        '--min_detection_confidence', type=float, default=0.7,
        help='手検出モデルの最小信頼値 [0.0, 1.0]'
    )
    parser.add_argument(
        '--min_tracking_confidence', type=float, default=0.5,
        help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]'
    )
    
    args = parser.parse_args()
    return args


def main() -> None:
    """メインループ"""
    args = get_args().__dict__
    capture_flag = False
    # setting camera device
    cap = cv2.VideoCapture(args['device'], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['height'])

    model_name = "hand_tracker"
    try:
        detector = hand_tracker.HandTracker(
            args['max_num_hands'],
            args['min_detection_confidence'],
            args['min_tracking_confidence']
        )
    except Exception as e:
        logger.error(e)
        exit(1)

    # main loop
    landmarks = []
    ds = detection_state.detection_state(detector=detector, cap=cap)
    print(type(detector))
    print(type(cap))
    while cap.isOpened():
        # 静止画またはカメラ入力
        ret, image = ds.cap.read()
        if not ret:
            break

        tmp_image = copy.deepcopy(image)
        tmp_landmark = []

        if detector.detect(image):
            tmp_image, tmp_landmark = detector.draw(tmp_image)
            # 手を検知して，欠損がないことを確認して手の検知数とランドマークを更新する．
            if (len(tmp_landmark)!=0 and len(tmp_landmark)%21 == 0):
                ds.landmark = np.array(tmp_landmark)
                ds.detected_hands_num = int(len(tmp_landmark)/21)
        
        if capture_flag==True:
            #sボタンを押せばデータの収集開始する
            get_datas.get_datas(ds)
            capture_flag = not capture_flag


        cv2.imshow('hand_tracker', cv2.resize(
            tmp_image, (args['width'], args['height'])))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            capture_flag = not capture_flag
        if key == ord('q'):
            break

    ds.cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
