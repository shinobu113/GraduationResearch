from copyreg import pickle
from json import load
from numpy import double
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

import detection_state
import matplotlib.pyplot as plt

X = []  # 説明変数
Y = []  # 目的変数
file_pathes = detection_state.get_file_path_list()

for file_path in file_pathes:
    ds = detection_state.load_detection_state(file_path)
    # x = [ds.operation_time]
    x = []
    x.extend(list(ds.joint_angle_mean['Left']))
    x.extend(list(ds.joint_angle_mean['Right']))
    label = ds.label
    X.append(x)
    Y.append(label)


def save_lasso_model(lasso):
    with open('lasso_model.pkl', 'wb') as f:
        pickle.dump(lasso, f)

def load_lasso_model(path):
    with open(path, 'rb') as f:
        lasso = pickle.load(f)
    return lasso

def model_fit(lasso_alpha: float = 1.0, threshold: float = 0.5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lasso = Lasso(alpha=lasso_alpha).fit(X_train, Y_train)
    X_test_predict = lasso.predict(X_test)
    X_test_predict = np.array(X_test_predict)#/max(X_test_predict) # 予想値を正規化
    X_test_predict = [1 if predict > threshold else 0 for predict in X_test_predict]
    save_lasso_model(lasso)

    X_test_predict = np.array(X_test_predict)
    Y_test = np.array(Y_test)

    match_cnt = list(X_test_predict == Y_test).count(True)
    acc = match_cnt/len(X_test_predict)
    tn, fp, fn, tp = confusion_matrix(Y_test, X_test_predict).ravel()
    # print(Y_test)
    # print(X_test_predict)
    # print(lasso.intercept_)
    return acc, [tn,fp,fn,tp], lasso.coef_, lasso.intercept_


# fig = plt.figure("Accuracy-Alpha Graph")
# ax = fig.add_subplot(1,1,1)
# ax.plot((np.array(range(1,101))/100), acc)
# plt.xlabel("Alpha", fontsize=15)
# plt.ylabel("Accuracy", fontsize=15)
# plt.show()

# mx = max(acc)
# ina = acc.index(mx)
# print(ina)


# mx = [0,0,0]
# for i in range(1, 101):
#     for j in range(1, 31):
#         res = [model_fit(lasso_alpha=0.01*i, threshold=0.5+(0.01*j))[0], i, j]
#         mx = [res[0], 0.01*i, 0.5+(0.01*j)] if mx[0] <= res[0] else mx

# print(mx)

# print(model_fit(lasso_alpha=1.0, threshold=0.62))
# x = [[133.7618819 , 158.0386377 , 166.4304732 , 121.70697591, 149.61480441, 169.22587927, 104.55543623, 137.8820008 , 99.954787  , 145.934889  , 117.15120198, 146.38841871, 145.14301347, 157.96377096, 158.62162379, 127.41866752,       120.18761217, 153.02368028, 106.91659358, 148.39772183,       107.88407104, 148.03522397, 124.00295339, 148.20250267]]
# lasso = load_lasso_model("lasso_model.pkl")