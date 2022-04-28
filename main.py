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


def get_args() -> argparse.Namespace:
    """引数取得

    Returns:
        argparse.Namespace: 取得結果
    """
    parser = argparse.ArgumentParser()

    # setting camera device
    parser.add_argument(
        "--device", help='デバイスカメラ:0  Webカメラ:1', type=int, default=0)
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
        '--max_num_hands', type=int, default=1, help='最大検出手数'
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
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    _, flame = cap.read()
    cv2.imshow("A", flame)
    cv2.waitKey(0)
    """args = get_args().__dict__

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
    while cap.isOpened():
        # 静止画またはカメラ入力
        if args['still_image'] == '':
            ret, image = cap.read()
            if not ret:
                break
            # image = cv2.flip(image, 1)
        else:
            image = cv2.imread(args['still_image'])
            if not exists(args['still_image']) and args['still_image'] != '':
                print("path error")
                break
            image = cv2.resize(image, (args['width'], args['height']))

        tmp_image = copy.deepcopy(image)

        if detector.detect(image):
            tmp_image, tmp_landmark = detector.draw(tmp_image)
            if len(tmp_landmark) == 21:
                landmarks.append(tmp_landmark)
            
        cv2.imshow(model_name, cv2.resize(
            tmp_image, (args['width'], args['height'])))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return"""


if __name__ == "__main__":
    main()
