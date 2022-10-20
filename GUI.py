from cProfile import label
from cgitb import text
from distutils import command
from importlib.resources import path
from logging import exception
from ntpath import join
from textwrap import fill
from turtle import width
from cv2 import FastFeatureDetector, FlannBasedMatcher, setIdentity
import numpy as np
import pickle
from calculation import calculate_joint_angle
import detection_state
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import threading as th
import hand_tracker
from loguru import logger
import copy
import math

lock = th.Lock()

class VideoPlayer(tk.Frame):
    def __init__(self,master=None):
        super().__init__(master,width=1000,height=500)
        self.master.resizable(width=False, height=False)
        self.config(bg="#000000")
        self.master.protocol("WM_DELETE_WINDOW", self.click_close)
        self.video = None
        self.playing = False
        self.video_thread = None
        self.gender = None 
        self.dominant_hand = None
        self.label = None
        self.ds = None
        self.detector = None
        self.mediapipe_flag = True
        self.path = None
        self.sub_window = None
        self.realtime_flag = False

        self.load_GUI_settings()
        if self.realtime_flag:
            self.gender = 'man'
            self.dominant_hand = 'Right'
            self.label = '1'
        else:
            self.load_detection_state()
        self.create_menu()
        

        # label = tk.Label(self.sub_window, text="パラメータ1")
        # label.grid(column=0, row=0, padx=10, pady=20)
        # scale = tk.Scale(self.sub_window, orient=tk.HORIZONTAL, showvalue=False)
        # scale.grid(column=1, row=0, padx=10, pady=20)

    def create_menu(self):
        #---------------------------------------
        #  ツールバー
        #---------------------------------------
        self.frame_menubar = tk.Frame(self.master,relief = tk.SUNKEN, bd = 2)
        button1 = tk.Button(self.frame_menubar, text = "ファイルの選択", command=self.open_filedialog)
        button2 = tk.Button(self.frame_menubar, text = "MediaPipe", command=self.push_mediapipe_button)
        button3 = tk.Button(self.frame_menubar, text = "リアルタイム", command=self.push_realtime_button)

        # ボタンをフレームに配置
        button1.pack(side = tk.LEFT)
        button2.pack(side = tk.LEFT)
        button3.pack(side = tk.LEFT)
        # ツールバーをウィンドの上に配置
        self.frame_menubar.pack(side=tk.TOP, fill=tk.X)

        #---------------------------------------
        #  作業者情報
        #---------------------------------------
        self.labelframe_operator = tk.LabelFrame(self.master, text="作業者情報", width=150, height=450)
        self.labelframe_operator.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.labelframe_operator.propagate(False)

        # 性別
        labelframe1 = tk.LabelFrame(self.labelframe_operator, text="性別", height=40)
        labelframe1.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox1 = ttk.Combobox(labelframe1, state="readonly", values=["man","woman"])
        self.combobox1.pack(padx=10, pady=10)
        self.combobox1.current(0 if self.gender=='man' else 1)


        # 利き手
        labelframe2 = tk.LabelFrame(self.labelframe_operator, text="利き手", height=40)
        labelframe2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox2 = ttk.Combobox(labelframe2, state="readonly", values=["Right","Left"])
        self.combobox2.pack(padx=10, pady=10)
        self.combobox2.current(0 if self.dominant_hand=='Right' else 1)

        # ラベル
        labelframe3 = tk.LabelFrame(self.labelframe_operator, text="ラベル", height=40)
        labelframe3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox3 = ttk.Combobox(labelframe3, state="readonly", values=["1", "0"])
        self.combobox3.pack(padx=10, pady=10)
        self.combobox3.current(0 if self.label==1 else 1)

        self.reload_button = tk.Button(self.labelframe_operator, command=self.push_reload_button ,text="Reload", height=5, font=("",20), relief=tk.GROOVE)
        self.reload_button.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)


        #---------------------------------------
        # パラメータ部
        #---------------------------------------
        self.labelframe_parameter = tk.LabelFrame(self.master, text="パラメータの調整", width=300, height=450)
        self.labelframe_parameter.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.labelframe_parameter.propagate(False)

        label_left = tk.Label(self.labelframe_parameter, text='左手')
        label_left.grid(row=0, column=1)
        label_right = tk.Label(self.labelframe_parameter, text='右手')
        label_right.grid(row=0, column=3)

        self.scale_values = []
        self.labels = []
        self.scales = []
        self.entries = []
        connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [0, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [9, 10, 11],
            [10, 11, 12],
            [13, 14, 15],
            [14, 15, 16],
            [17, 18, 19],
            [18, 19, 20]
        ]
        parameter_name = ['0-1-2','1-2-3','2,3,4','0-5-6','5-6-7','6-7-8','9-10-11','10-11-12','11-12-13','14-15-16','17-18-19','18-19-20']
        for i in range(12):
            scale_value1 = tk.IntVar()
            scale_value2 = tk.IntVar()
            try:
                scale_value1.set(int(self.ds.joint_angle_mean['Left'][i]))
                scale_value2.set(int(self.ds.joint_angle_mean['Right'][i]))
            except:
                scale_value1.set(0)
                scale_value2.set(0)

            self.scale_values.append([scale_value1, scale_value2])

            label = tk.Label(self.labelframe_parameter, text=parameter_name[i])
            self.labels.append(label)
            label.grid(row=i+1, column=0, padx=10, pady=10)
            
            entry1 = tk.Entry(self.labelframe_parameter, textvariable=self.scale_values[i][0], width=3, state=tk.DISABLED)
            entry1.grid(row=i+1, column=2, padx=10)
            entry2 = tk.Entry(self.labelframe_parameter, textvariable=self.scale_values[i][1], width=3, state=tk.DISABLED)
            entry2.grid(row=i+1, column=4, padx=10)

            scale1 = tk.Scale(self.labelframe_parameter, to=180, variable=self.scale_values[i][0] ,orient=tk.HORIZONTAL, width=10, showvalue=False, length=150)
            scale1.grid(row=i+1, column=1, padx=10, pady=10)
            scale2 = tk.Scale(self.labelframe_parameter, to=180, variable=self.scale_values[i][1] ,orient=tk.HORIZONTAL, width=10, showvalue=False, length=150)
            scale2.grid(row=i+1, column=3, padx=10, pady=10)
            self.scales.append([scale1, scale2])
            
        
        #---------------------------------------
        # 動画再生
        #---------------------------------------
        self.labelframe_video = tk.LabelFrame(self.master, text="動画再生", width=810, height=620)
        self.labelframe_video.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_button = tk.Button(self.labelframe_video, command=self.push_play_button)
        self.video_button.pack(fill=tk.BOTH)
        self.labelframe_video.propagate(False)
            


    def push_mediapipe_button(self):
        self.mediapipe_flag = not self.mediapipe_flag


    def push_realtime_button(self):
        self.realtime_flag = not self.realtime_flag
        self.video.release()

        if self.realtime_flag:
            with open('GUI_settings.txt', 'w') as f:
                f.write(f'-1 -1')
            self.get_video()
            self.load_GUI_settings()
        else:
            self.open_filedialog()


    def click_close(self):
        # 現在開いているファイルをGUI_settings.txtに上書きして保存する．
        with open('GUI_settings.txt', 'w') as f:
            f.write(f'{self.folder_name} {self.file_name}')
        self.master.destroy()


    def load_GUI_settings(self):
        # GUIの設定を読み込む
        try:
            with open("./GUI_settings.txt", "r") as f:
                tmp = f.read().split()
                self.folder_name = tmp[0]
                self.file_name = tmp[1]
            # MediaPipeの検出器を作成
            self.detector = hand_tracker.HandTracker(
                2, 0.7, 0.5
            )
            # フォルダ名とファイル名が-1のときはリアルタイム処理とする
            if self.folder_name==str(-1) and self.file_name==str(-1):
                self.master.title("Real Time")
                self.realtime_flag = True
                self.get_video()
            else:
                self.master.title(f'ファイル：./data/{self.folder_name}/{self.file_name}.mp4')
                self.path = f'./data/{self.folder_name}/{self.file_name}.mp4'
                self.get_video(self.path)
        except Exception as e:
            print('例外発生です')
            print(e)
    

    def load_detection_state(self):
        # 動画を解析したPKL形式のファイルを読み込む
        try:
            with open(f'./data/{self.folder_name}/{self.file_name}.pkl', 'rb') as f:
                self.ds = pickle.load(f)
                self.gender         = self.ds.gender
                self.dominant_hand  = self.ds.dominant_hand
                self.label          = self.ds.label
        except Exception as e:
            print('例外発生です')
            logger.error(e)


    def open_filedialog(self):
        try:
            self.filename = filedialog.askopenfilename(
                title = "ファイルの選択",
                # filetypes = [("PKL", ".pkl"), ("MP4", ".mp4"),("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
                initialdir = "./data/" # 自分自身のディレクトリ
            )
            _split = self.filename.split('/')
            self.folder_name = _split[-2]
            self.file_name = _split[-1].split('.')[0]
        except Exception as e:
            print('ファイルが選択されませんでした')
        
        with open('GUI_settings.txt', 'w') as f:
            f.write(f'{self.folder_name} {self.file_name}')


        self.get_video(f'./data/{self.folder_name}/{self.file_name}.mp4')
        self.load_GUI_settings()
        self.load_detection_state()
        
        self.combobox1.current(0 if self.gender=='man' else 1)
        self.combobox2.current(0 if self.dominant_hand=='Right' else 1)
        self.combobox3.current(0 if self.label==1 else 1)


    def get_video(self, path=""):
        if path=="":
            self.video = cv2.VideoCapture(1)
        else:
            self.video = cv2.VideoCapture(path)


    def push_reload_button(self):
        try:
            self.gender = self.combobox1.get()
            self.dominant_hand = self.combobox2.get()
            self.label = self.combobox3.get()
            self.ds.gender = self.gender
            self.ds.dominant_hand = self.dominant_hand
            self.ds.label = self.label
            with open(f'./data/{self.folder_name}/{self.file_name}.pkl', 'wb') as f:
                pickle.dump(self.ds, f)

        except Exception as e:
            print('例外発生です')
            print(e)


    def push_play_button(self):
        if self.video == None:
            messagebox.showerror('エラー','動画データがありません')
            return
        self.playing = not self.playing

        if self.playing:
            self.video_thread = th.Thread(target=self.video_frame_timer)
            self.video_thread.setDaemon(True)
            self.video_thread.start()          
        else:
            self.video_thread = None 


    def next_frame(self):
        global lock
        lock.acquire()
        ret, self.frame = self.video.read()

        if not ret:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        else:
            tmp_image = copy.deepcopy(self.frame)
            if self.detector.detect(self.frame):
                tmp_image, tmp_landmark_dict = self.detector.draw(tmp_image)
                joint_angle = self.calculate_joint_angle_1frame(tmp_landmark_dict)

                for i in range(12):
                    if joint_angle['Left'] != []:
                        self.scale_values[i][0].set(int(joint_angle['Left'][i]))
                    if joint_angle['Right'] != []:
                        self.scale_values[i][1].set(int(joint_angle['Right'][i]))

            rgb = cv2.cvtColor(tmp_image if self.mediapipe_flag else self.frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            image = ImageTk.PhotoImage(pil)

            self.video_button.config(image=image)
            self.video_button.image = image
        lock.release()


    def calculate_joint_angle_1frame(self, landmark):
        connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [0, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [9, 10, 11],
            [10, 11, 12],
            [13, 14, 15],
            [14, 15, 16],
            [17, 18, 19],
            [18, 19, 20]
        ]

        joint_angle = {'Left':[], 'Right':[]}

        for hand in ['Left', 'Right']:
            # 手を検出していないときはスルー
            if landmark[hand]==[]:
                continue
            
            angles = []
            # 関節のつながりからなす角度を計算する
            for connection in connections:
                A = np.array(landmark[hand][connection[0]])
                B = np.array(landmark[hand][connection[1]])
                C = np.array(landmark[hand][connection[2]])
                BA = A-B
                BC = C-B
                norm_AB = np.linalg.norm(BA)
                norm_BC = np.linalg.norm(BC)
                inner_AB_BC = np.inner(BA, BC)
                angle_rad = np.arccos(inner_AB_BC/(norm_AB*norm_BC))
                angle_deg = math.degrees(angle_rad)
                angles.append(angle_deg)
            joint_angle[hand] = angles
        return joint_angle


    def video_frame_timer(self):
        while self.playing:
            self.next_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(master=root)
    root.mainloop()