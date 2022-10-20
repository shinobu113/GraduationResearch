from turtle import distance
from numpy import double
import numpy as np

import detection_state
import matplotlib.pyplot as plt
from scipy import stats
from pprint import pprint

from sklearn.neighbors import DistanceMetric

def calc_EuclidDistance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)



def main():
    #------------------------
    # データの作成部分
    #------------------------
    files = detection_state.get_file_path_list()
    data = []
    for file in files:
        ds = detection_state.load_detection_state(file)
        joint_angle_mean = []
        for x in [4,6,10]:
            joint_angle_mean.append(ds.joint_angle_mean['Right'][x])
        data.append([joint_angle_mean, ds.label])
    
    distances = np.array([np.inf for i in range((len(files)**2)*2)])
    distances = distances.reshape(len(files), len(files), 2)

    for i in range(len(files)):
        for j in range(len(files)):
            dis = calc_EuclidDistance(data[i][0], data[j][0])
            if i == j:
                dis = np.inf
            distances[i][j] = (dis, j)
    # pprint(sorted(distances[0], key=lambda x:x[0]))
    
    # distances[i][j]はiとjのjoint_angle_meanの距離とソート用のインデックスを保持している
    distances = [sorted(dis, key=lambda x:x[0]) for dis in distances]
    

    #----------------------
    # 距離が短いk個を取り出し異常度を決定する
    #----------------------
    # https://qiita.com/ngayope330/items/fc941b8d49b90319748e
    # https://rpubs.com/tintstyle/classification2
    # https://dev.classmethod.jp/articles/2017ad_20171218_knn/
    k = 3
    result = []
    for i in range(len(files)):
    
    # i番目を近傍k個の多数決で判定する
    # for i in range(5):
        label = 0
        for j in range(k):
            # i番目に最も距離の近いデータのindex
            index = int(distances[i][:k][j][1])
            label += data[index][1]
        result.append([1 if label/k >= 0.5 else 0, data[i][1]])
    
    temp = [1 if res[0]==res[1] else 0 for res in result]
    sum = 0
    for i in temp:
        sum += i
    print(sum/len(temp))

        


            


        


if __name__=="__main__":
    main()