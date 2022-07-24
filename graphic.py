from distutils.command.build_clib import show_compilers
from cv2 import line
from matplotlib import animation
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d
import os
import glob
from pathlib import Path

import detection_state


class Graphic_3D():
    """
    リアルタイムではなく計算終わったランドマークを3Dで表示する．
    """
    connections = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [5, 9, 13, 17, 0, 5]
    ]
    def animate(self, cnt :int):

        for i, hand in enumerate(['Left', 'Right']):
            landmark = np.array(self.landmarks[cnt][hand])
            if landmark.size == 0: # 空の配列ならスルー
                continue
            x = landmark[:,0]
            y = landmark[:,1]
            z = landmark[:,2]

            x_min = min(x)
            y_min = min(y)
            z_min = min(z)
            x_max = max(x)
            y_max = max(y)
            z_max = max(z)
            scale = 1.2

            if hand == 'Left':
                self.ax1.cla()
                self.ax1.set_title('Left Hand')
                self.ax1.set_xlim(x_min/scale, x_max*scale)
                self.ax1.set_ylim(y_min/scale, y_max*scale)
                self.ax1.set_zlim(z_min/scale, z_max*scale)
                self.ax1.scatter(x,y,z,c=self.color_dict[hand])
                for connection in self.connections:
                    X = []
                    Y = []
                    Z = []
                    for conn in connection:
                        X.append(x[conn])
                        Y.append(y[conn])
                        Z.append(z[conn])
                    line = art3d.Line3D(X, Y, Z, color=self.color_dict[hand], linewidth=6)
                    self.ax1.add_line(line)



            else:
                self.ax2.cla()
                self.ax2.set_title('Right Hand')
                self.ax2.set_xlim(x_min/scale, x_max*scale)
                self.ax2.set_ylim(y_min/scale, y_max*scale)
                self.ax2.set_zlim(z_min/scale, z_max*scale)
                self.ax2.scatter(x,y,z,c=self.color_dict[hand])
                for connection in self.connections:
                    X = []
                    Y = []
                    Z = []
                    for conn in connection:
                        X.append(x[conn])
                        Y.append(y[conn])
                        Z.append(z[conn])
                    line = art3d.Line3D(X, Y, Z, color=self.color_dict[hand], linewidth=6)
                    self.ax2.add_line(line)
    
    def plot(self, animation_path='./animation.gif'):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.landmarks),interval=100, blit=False)
        ani.save(animation_path, writer="ffmpeg")
        plt.show()

    def __init__(self, landmarks :list) -> None:
        # インスタンス変数として定義する必要があることに注意
        self.landmarks = landmarks
        self.color_dict = {'Left':'blue', 'Right':'green'}
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1, projection=Axes3D.name)
        self.ax2 = self.fig.add_subplot(1, 2, 2, projection=Axes3D.name)
        
        


def show_joint_angles() -> None:
    """
    右手と左手の関節の角度データをグラフで表示する
    """

    # グラフ領域の設定1
    fig1 = plt.figure("Left Hand", figsize=(12,6))
    fig1.subplots_adjust(wspace=0.2, hspace=0.5)
    fig1.suptitle("Left Hand")
    _axes1 = []
    for i in range(12):
        ax = fig1.add_subplot(3,4,i+1)
        ax.set_title(str(i+1))
        _axes1.append(ax)
    
    # グラフ領域の設定2
    fig2 = plt.figure("Right Hand", figsize=(12,6))
    fig2.subplots_adjust(wspace=0.2, hspace=0.5)
    fig2.suptitle("Right Hand")
    _axes2 = []
    for i in range(12):
        ax = fig2.add_subplot(3,4,i+1)
        ax.set_title(str(i+1))
        _axes2.append(ax)


    BASE_DIR_PATH = './data/'
    dir_names = os.listdir(BASE_DIR_PATH)
    for dir_name in dir_names:
        video_paths = glob.glob(f'{BASE_DIR_PATH}/{dir_name}/*.mp4')
        
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).stem  # 拡張子抜きのファイル名
            
            if video_name=='original':  # 動画の元データには解析を行わない
                continue

            input_video_path = video_path
            output_pkl_path  = f'{BASE_DIR_PATH}/{dir_name}/{video_name}.pkl'
            
            if os.path.isfile(output_pkl_path):
                ds = detection_state.load_detection_state(pkl_path=output_pkl_path)
            else:
                continue
            
            left_angles = []
            right_angles = []
            for joint_angle in ds.joint_angles:
                left_angles.append(joint_angle['Left'])
                right_angles.append(joint_angle['Right'])
            left_angles = np.array(left_angles)
            right_angles = np.array(right_angles)

            for i in range(12):
                _axes1[i].plot(left_angles[:,i])
                _axes2[i].plot(right_angles[:,i])
    plt.show()



def show_operation_time_hist() -> None:
    """
    作業時間の分布をヒストグラムで表示
    """
    fig = plt.figure("作業時間分布")
    ax = fig.add_subplot(1,1,1)

    BASE_DIR_PATH = './data/'
    dir_names = os.listdir(BASE_DIR_PATH)
    operation_times = []
    for dir_name in dir_names:
        video_paths = glob.glob(f'{BASE_DIR_PATH}/{dir_name}/*.mp4')
        
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).stem  # 拡張子抜きのファイル名
            
            if video_name=='original':  # 動画の元データには解析を行わない
                continue

            input_video_path = video_path
            output_pkl_path  = f'{BASE_DIR_PATH}/{dir_name}/{video_name}.pkl'
            
            if os.path.isfile(output_pkl_path):
                ds = detection_state.load_detection_state(pkl_path=output_pkl_path)
            else:
                continue
            operation_times.append(ds.operation_time)

    ax.hist(operation_times, bins=20)
    plt.show()

show_operation_time_hist()