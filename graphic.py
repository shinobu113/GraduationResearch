from cv2 import line
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d
import detection_state
# https://sabopy.com/py/matplotlib-animation-78/

# ワイヤーフレームやってみたい！！

class Graphic_3D():
    """
    リアルタイムではなく計算終わったランドマークを3Dで表示する．
    """
    landmarks :list
    color_dict = {'Left':'blue', 'Right':'green'}
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection=Axes3D.name)
    ax2 = fig.add_subplot(1, 2, 2, projection=Axes3D.name)
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
    
    def __init__(self, landmarks :list) -> None:
        self.landmarks = landmarks
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.landmarks),interval=100, blit=False)
        plt.show()
        
        
