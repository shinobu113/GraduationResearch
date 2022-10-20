from numpy import double
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

import detection_state
import matplotlib.pyplot as plt

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


def model_fit(lasso_alpha: float = 1.0, threshold: float = 0.5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lasso = Lasso(alpha=lasso_alpha).fit(X_train, Y_train)
    X_test_predict = lasso.predict(X_test)
    X_test_predict = np.array(X_test_predict)/max(X_test_predict) # 予想値を正規化
    X_test_predict = [1 if predict > threshold else 0 for predict in X_test_predict]

    X_test_predict = np.array(X_test_predict)
    Y_test = np.array(Y_test)

    match_cnt = list(X_test_predict == Y_test).count(True)
    acc = match_cnt/len(X_test_predict)
    tn, fp, fn, tp = confusion_matrix(Y_test, X_test_predict).ravel()
    # print(Y_test)
    # print(X_test_predict)
    print(lasso.intercept_)
    return acc, [tn,fp,fn,tp], lasso.coef_


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

print(model_fit(lasso_alpha=1.0, threshold=0.62))