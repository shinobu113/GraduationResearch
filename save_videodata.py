"""
普通に動画撮るソフト使えばいいので必要ない
videocaptureにDSHOWを指定すると動画が保存できないが使わないと少しカクつく
"""
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



def main():
    args = get_args().__dict__

    cap = cv2.VideoCapture(args['device'])
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (640, 480)
    video = cv2.VideoWriter('outvideo.mp4', fmt, fps, size)

    try:
        detector = hand_tracker.HandTracker(
            args['max_num_hands'],
            args['min_detection_confidence'],
            args['min_tracking_confidence']
        )
    except Exception as e:
        logger.error(e)
        exit(1)

    while cap.isOpened():

        ret, image = cap.read()
        if not ret:
            break

        tmp_image = copy.deepcopy(image)
        if detector.detect(image):
            tmp_image, _ = detector.draw(tmp_image)
        
        cv2.imshow('hand_tracker', tmp_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # 動画保存処理
            while(1):
                ret, image = cap.read()
                video.write(image)# フレーム画像を動画として保存
                
                tmp_image = copy.deepcopy(image)
                if detector.detect(image):
                    tmp_image, _ = detector.draw(tmp_image)
                cv2.circle(tmp_image, (35, 35), 20, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                
                cv2.imshow('hand_tracker', tmp_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('w'):
                    break
                
    video.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()