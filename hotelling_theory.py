from numpy import double
import numpy as np

import detection_state
import matplotlib.pyplot as plt
from scipy import stats
from pprint import pprint

files = detection_state.get_file_path_list()
# ラベルが1のデータのみを集める
label1_list = [file for file in files if detection_state.load_detection_state(file).label==1]

joint_angles = []
for file in label1_list:
    ds = detection_state.load_detection_state(file)
    joint_angles.append(
        [ds.joint_angle_mean['Right'][4], ds.joint_angle_mean['Right'][6], ds.joint_angle_mean['Right'][10]]
    )

joint_angles = np.array(joint_angles)
joint_angle_mean = np.mean(joint_angles, axis=0)
joint_angle_variance = np.var(joint_angles, axis=0)

data = [detection_state.load_detection_state(file).joint_angle_mean['Right'][6] for file in files]

anomaly_scores = [(x-joint_angle_mean[1])**2 / joint_angle_variance[1] for x in data]

threshold = stats.chi2.interval(0.75, 1)[1]
plt.plot(range(len(files)), anomaly_scores, "o", color = "b")
plt.plot([0,len(files)],[threshold, threshold], 'k-', color = "r", ls = "dashed")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.show()