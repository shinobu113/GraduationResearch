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
import detection_state
import graphic

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
    # cap = cv2.VideoCapture(args['device'], cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(args['device'])
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
    ds = detection_state.DetectionState()
    while cap.isOpened():
        # 静止画またはカメラ入力
        ret, image = cap.read()
        if not ret:
            print("Break")
            break

        tmp_image = copy.deepcopy(image)
        tmp_landmark = []

        if detector.detect(image):
            tmp_image, tmp_landmark_dict = detector.draw(tmp_image)
            ds.update_landmarks(tmp_landmark_dict)

        cv2.imshow('hand_tracker', cv2.resize(tmp_image, (args['width'], args['height'])))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite('./figures/screenshot.png', tmp_image)
        if key == ord('q'):
            break
    
    gra = graphic.Graphic_3D(ds.landmarks)
    # gra.plot()
    
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
