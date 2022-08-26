from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=0)

# lasso = Lasso(alpha=1).fit(X_train, Y_train)
# print(lasso.coef_)

# X_test_predict = lasso.predict(X_test)
# X_test_predict = [1 if predict > 0.5 else 0 for predict in X_test_predict]

# X_test_predict = np.array(X_test_predict)
# Y_test = np.array(Y_test)

# match_cnt = list(X_test_predict == Y_test).count(True)
# print(match_cnt/len(X_test_predict))


acc = []
for i in range(1, 101):
    print(i)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
    lasso = Lasso(alpha=0.01*i).fit(X_train, Y_train)
    X_test_predict = lasso.predict(X_test)
    X_test_predict = [1 if predict > 0.5 else 0 for predict in X_test_predict]

    X_test_predict = np.array(X_test_predict)
    Y_test = np.array(Y_test)

    match_cnt = list(X_test_predict == Y_test).count(True)
    acc.append(match_cnt/len(X_test_predict))

print(acc)
print(max(acc))

fig = plt.figure("Accuracy-Alpha Graph")
ax = fig.add_subplot(1,1,1)

ax.plot((np.array(range(1,101))/100), acc)
plt.xlabel("Alpha", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.show()