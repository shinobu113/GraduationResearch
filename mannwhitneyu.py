from scipy import stats
from fileinput import filename
import os
import sys
import argparse
from pprint import pprint
from time import time, sleep
import glob
from turtle import color
from cv2 import FILE_NODE_MAP, destroyWindow, sqrt
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
import random

files = detection_state.get_file_path_list()
b0 = []
b1 = []
for file in files:
    ds = detection_state.load_detection_state(file)
    if ds.label==0:
        b0.append(ds.lasso_mean_value)
    else:
        b1.append(ds.lasso_mean_value)
b1 = np.array(b1)
b0 = np.array(b0)
# b1 = random.sample(b1, 30)
# b1 = b1[:20]
# b0 = b0[:20]
# print(n1, n2)
# b1 = [50,49,49,47,45,45,43,43,43,41,38,37,36,35,34,33,33,31,31,30]
# b0 = [48,48,46,44,42,42,42,41,41,36,34,32,30,30,29,28,28,28,26,25,24,22]
# b1 = [50, 49, 49, 47, 45, 45, 43, 43, 43, 41, 38, 37, 36, 35, 34, 33, 33, 31, 31, 30]
# b0 = [48, 48, 46, 44, 42, 42, 42, 41, 41, 36, 34, 32, 30, 30, 29, 28, 28, 28, 26, 25, 24, 22]
n1 = len(b0)
n2 = len(b1)
result = stats.mannwhitneyu(b0,b1, alternative='two-sided')
U = result.statistic
U = 700
m = (n1*n2)/2
sigma = ((n1*n2*(n1+n2+1))/12)**0.5
z = round((U-m)/sigma, 2)
print(f'n1={n1}\nn2={n2}\nU={result.statistic}\npvalue={result.pvalue}\nm={m}\nsigma={sigma}\nz={z}')