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
l = []
a = [4,6,10]
for file in files:
    ds = detection_state.load_detection_state(file)
    # l.append([[ds.joint_angle_mean['Left'][4],ds.joint_angle_mean['Left'][6],ds.joint_angle_mean['Left'][10]],
    #         [ds.joint_angle_mean['Right'][4],ds.joint_angle_mean['Right'][6],ds.joint_angle_mean['Right'][10]],[ds.label,0,0]])
    l.append([[ds.joint_angle_mean['Left'][a[0]],ds.joint_angle_mean['Left'][a[1]],ds.joint_angle_mean['Left'][a[2]]],
            [ds.joint_angle_mean['Right'][a[0]],ds.joint_angle_mean['Right'][a[1]],ds.joint_angle_mean['Right'][a[2]]],[ds.label,0,0]])

label0 = [i for i in l if i[2][0]==0]
label1 = [i for i in l if i[2][0]==1]
label0 = np.array(label0)
label1 = np.array(label1)

fig = plt.figure(figsize=(12, 6))
axes = [fig.add_subplot(1,6,i+1) for i in range(6)]
fig.subplots_adjust(wspace=0.8, hspace=0.2)

bpes = [axes[i].boxplot([label0[:,int(i/3),(i%3)], label1[:,int(i/3),(i%3)]], patch_artist=True, labels=['0', '1'], widths=0.3, medianprops=dict(color='black', linewidth=1)) for i in range(6)]
titles = ['L-5-6-7', 'L-9-10-11', 'L-17-18-19', 'R-5-6-7', 'R-9-10-11', 'R-17-18-19']

for ax, title in zip(axes, titles):
    ax.set_title(title)
colors=['pink', 'lightblue']

for bp in bpes:
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

plt.show()
