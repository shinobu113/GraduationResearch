from turtle import color
from numpy import double
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import detection_state


X = []  # 説明変数
Y = []  # 目的変数
file_pathes = detection_state.get_file_path_list()

for file_path in file_pathes:
    ds = detection_state.load_detection_state(file_path)
    x = [ds.operation_time]
    x.extend(list(ds.joint_angle_mean['Left']))
    x.extend(list(ds.joint_angle_mean['Right']))
    label = ds.label
    X.append(x)
    Y.append(label)


def model_fit(lasso_alpha: float = 1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lasso = Lasso(alpha=lasso_alpha).fit(X_train, Y_train)
    X_test_predict = lasso.predict(X_test)
    
    return  Y_test, X_test_predict

Y_test, X_test_predict = model_fit()

fpr, tpr, thresholds = roc_curve(Y_test, X_test_predict, drop_intermediate=False)

plt.plot(fpr, tpr, marker='o', linewidth=3)
plt.xlabel('FPR: False positive rate', fontsize=18)
plt.ylabel('TPR: True positive rate', fontsize=18)
plt.grid()
plt.savefig('./figures/sklearn_roc_curve222.png')

print(roc_auc_score(Y_test, X_test_predict))

# NG_data = []
# OK_data = []
# for i, j in zip(Y_test, X_test_predict):
#     if i == 0:
#         NG_data.append([i,j])
#     else:
#         OK_data.append([i,j])

# plt.hist(np.array(OK_data)[:,1], color="blue")    
# plt.hist(np.array(NG_data)[:,1], color="red")
# # plt.hist(np.array(OK_data)[:,1], color="blue")

# plt.show()