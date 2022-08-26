from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

import detection_state

file_pathes = detection_state.get_file_path_list()

x = []
y = []
for file_path in file_pathes:
    ds = detection_state.load_detection_state(file_path)
    a = list(ds.joint_angle_mean['Left'])
    a = [ds.operation_time]
    a.extend(list(ds.joint_angle_mean['Right']))
    x.append(a)
    y.append(ds.label)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)
# print(lasso.score(X_test, y_test))
# print(lasso.predict(X_test))
# print(f"training dataに対しての精度: {lasso.score(X_train, y_train):.2}")
# print(f"test set scoreに対しての精度: {lasso.score(X_test, y_test):.2f}")

temp = lasso.predict(x)

z = [[i] for i in temp]
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(z, y, random_state=0)
model = SVC(gamma='scale')
model.fit(X_train, y_train)

res = model.predict(z)
print(y)
print(model.predict(z))

cnt = 0
for i,yy in enumerate(y):
    if yy == res[i]:
        cnt+=1

print(cnt/len(y))
