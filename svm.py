from sklearn import svm

import detection_state


file_pathes = detection_state.get_file_path_list()

x = []
y = []
for file_path in file_pathes:
    ds = detection_state.load_detection_state(file_path)
    a = list(ds.joint_angle_mean['Left'])
    a.extend(list(ds.joint_angle_mean['Right']))
    x.append(a)
    y.append(ds.label)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.8, random_state=0)


from sklearn.svm import SVC
model = SVC(gamma='scale')
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(Y_test)
print(Y_pred)

# 線形分離とかが関係しているかも？？