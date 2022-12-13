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


    
files = detection_state.get_file_path_list()
b0 = []
b1 = []
for file in files:
    ds = detection_state.load_detection_state(file)
    # l.append([[ds.joint_angle_mean['Left'][4],ds.joint_angle_mean['Left'][6],ds.joint_angle_mean['Left'][10]],
    #         [ds.joint_angle_mean['Right'][4],ds.joint_angle_mean['Right'][6],ds.joint_angle_mean['Right'][10]],[ds.label,0,0]])
    if ds.label==0:
        b0.append(ds.lasso_mean_value)
    else:
        b1.append(ds.lasso_mean_value)

a1 = np.var(b1)
a2 = np.var(b0)
print(a1*100, a2*100)
fig = plt.figure()
plt.hist(b1)
plt.show()

