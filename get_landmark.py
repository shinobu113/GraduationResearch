import os
import argparse
from pprint import pprint
from time import time, sleep
import glob
from cv2 import FILE_NODE_MAP
# import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
from loguru import logger

import hand_tracker

# 関数名変えよう
def analyze(video_path :str)->None:
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, flame = cap.read()
        if not ret:
            break
        cv2.imshow("A", flame)
        key = cv2.waitKey(1) & 0xFF #waitkeyがないとエラーが出る
        if key == ord('q'):
            break
    print("---------------------------")


def main():
    
    FOLDER_PATH = './data/'
    folders = os.listdir(FOLDER_PATH)
    
    for folder in folders:
        video_path = glob.glob(f'{FOLDER_PATH}/{folder}/*.mp4')
        analyze(video_path[0])




if __name__ == "__main__":
    main()