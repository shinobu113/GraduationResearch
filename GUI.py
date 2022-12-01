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
import lasso
import calculation
import detection_state
from tkinter import BooleanVar, DoubleVar, filedialog
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
from numpy import pad
from pathlib import Path
import os

from pyparsing import col

lock = th.Lock()

class VideoPlayer(tk.Frame):
    def __init__(self,master=None):
        super().__init__(master,width=1000,height=500)
        self.master.resizable(width=False, height=False)
        self.config(bg="#000000")
        self.master.protocol("WM_DELETE_WINDOW", self.click_close_mainwindow)
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
        self.realtime_flag = False
        self.realtime_lasso_predict = DoubleVar()
        self.lasso_predict_mean = DoubleVar()
        self.lasso_predict_sum = 0 #Lassoの予測値の合計
        self.frame_cnt = 0 #スタートからのフレーム数
        self.threshold = 0.62 
        self.is_left_handed = BooleanVar()
        self.is_left_handed.set(False)
        self.is_capture_started = False
        self.lasso_mean_value = 0.0


        self.load_GUI_settings()
        self.lasso = lasso.load_lasso_model('lasso_model.pkl')
        if self.realtime_flag:
            self.gender = 'man'
            self.dominant_hand = 'Right'
            self.label = '1'
        else:
            self.load_detection_state()
        self.create_menu()

    def create_menu(self):
        #---------------------------------------
        #  ツールバー
        #---------------------------------------
        self.frame_menubar = tk.Frame(self.master,relief = tk.SUNKEN, bd = 2)
        button1 = tk.Button(self.frame_menubar, text = "ファイルの選択", command=self.open_filedialog, font=("MS Pゴシック", 10, "bold"))
        button2 = tk.Button(self.frame_menubar, text = "ファイルの解析", command=self.push_file_analyze_button, font=("MS Pゴシック", 10, "bold"))
        button3 = tk.Button(self.frame_menubar, text = "MediaPipe", command=self.push_mediapipe_button, font=("MS Pゴシック", 10, "bold"))
        button4 = tk.Button(self.frame_menubar, text = "リアルタイム", command=self.push_realtime_button, font=("MS Pゴシック", 10, "bold"))
        button5 = tk.Button(self.frame_menubar, text = "検出情報出力", command=self.push_output_button, font=("MS Pゴシック", 10, "bold"))
        button6 = tk.Button(self.frame_menubar, text = "動画撮影", command=self.push_movie_button, font=("MS Pゴシック", 10, "bold"))

        # ボタンをフレームに配置
        button1.pack(side = tk.LEFT)
        button2.pack(side = tk.LEFT)
        button3.pack(side = tk.LEFT)
        button4.pack(side = tk.LEFT)
        button5.pack(side = tk.LEFT)
        button6.pack(side = tk.LEFT)
        # ツールバーをウィンドの上に配置
        self.frame_menubar.pack(side=tk.TOP, fill=tk.X)

        #---------------------------------------
        #  作業者情報
        #---------------------------------------
        self.labelframe_operator = tk.LabelFrame(self.master, text="作業者情報", width=150, height=450, font=("MS Pゴシック", 10, "bold"))
        self.labelframe_operator.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.labelframe_operator.propagate(False)

        # 性別
        labelframe1 = tk.LabelFrame(self.labelframe_operator, text="性別", height=40, font=("MS Pゴシック", 10, "bold"))
        labelframe1.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox1 = ttk.Combobox(labelframe1, state="readonly", values=["man","woman"])
        self.combobox1.pack(padx=10, pady=10)
        self.combobox1.current(0 if self.gender=='man' else 1)

        # 利き手
        labelframe2 = tk.LabelFrame(self.labelframe_operator, text="利き手", height=40, font=("MS Pゴシック", 10, "bold"))
        labelframe2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox2 = ttk.Combobox(labelframe2, state="readonly", values=["Right","Left"])
        self.combobox2.pack(padx=10, pady=10)
        self.combobox2.current(0 if self.dominant_hand=='Right' else 1)

        # ラベル
        labelframe3 = tk.LabelFrame(self.labelframe_operator, text="ラベル", height=40, font=("MS Pゴシック", 10, "bold"))
        labelframe3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox3 = ttk.Combobox(labelframe3, state="readonly", values=["1", "0"])
        self.combobox3.pack(padx=10, pady=10)
        self.combobox3.current(0 if self.label=='1' else 1)

        # 左利き用に動画を反転するかどうか
        labelframe4 = tk.LabelFrame(self.labelframe_operator, text="左利き用", height=40, font=("MS Pゴシック", 10, "bold"))
        labelframe4.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.checkbutton = tk.Checkbutton(labelframe4, text='左利きか否か', variable=self.is_left_handed, font=("MS Pゴシック", 10, "bold"))
        self.checkbutton.pack(padx=10, pady=10)

        # 作業者情報の変更の保存ボタン
        reload_button = tk.Button(self.labelframe_operator, text='作業者情報更新', font=("MS Pゴシック", 10, "bold"), command=self.push_reload_button, height=5)
        reload_button.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        #---------------------------------------
        # パラメータ部
        #---------------------------------------
        self.labelframe_parameter = tk.LabelFrame(self.master, text="パラメータ", width=300, height=450, font=("MS Pゴシック", 10, "bold"))
        self.labelframe_parameter.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.labelframe_parameter.propagate(False)

        # Lassoのリアルタイム出力
        label_lasso_realtime = tk.Label(self.labelframe_parameter, text='Lassoのリアルタイム予測値', font=("MS Pゴシック", 10, "bold"))
        label_lasso_realtime.grid(row=0, column=0, columnspan=2, pady=15)
        scale_lasso_realtime = tk.Scale(self.labelframe_parameter, variable=self.realtime_lasso_predict,
                                resolution=0.01, to=1.5, orient=tk.HORIZONTAL, width=15, showvalue=False, length=180)
        scale_lasso_realtime.grid(row=0, column=3, pady=15)
        entry_lasso_realtime_value = tk.Entry(self.labelframe_parameter, textvariable=self.realtime_lasso_predict, width=4, state=tk.DISABLED)
        entry_lasso_realtime_value.grid(row=0, column=4, pady=15)
        
        self.lasso_canvas_realtime = tk.Canvas(self.labelframe_parameter, width=20, height=20)
        self.lasso_canvas_realtime.grid(row=0, column=2)
        self.lasso_canvas_realtime.create_oval(2,2,20,20, width=0, fill='#ff0000', tag='circle_realtime')
        
        # Lassoの予測値の平均用
        label_lasso_mean = tk.Label(self.labelframe_parameter, text='Lassoの予測値の平均', font=("MS Pゴシック", 10, "bold"))
        label_lasso_mean.grid(row=1, column=0, columnspan=2, pady=15)
        scale_lasso_mean = tk.Scale(self.labelframe_parameter, variable=self.lasso_predict_mean,
                                resolution=0.01, to=1.5, orient=tk.HORIZONTAL, width=15, showvalue=False, length=180)
        scale_lasso_mean.grid(row=1, column=3, pady=15)
        entry_lasso_mean_value = tk.Entry(self.labelframe_parameter, textvariable=self.lasso_predict_mean, width=4, state=tk.DISABLED)
        entry_lasso_mean_value.grid(row=1, column=4, pady=15)

        self.lasso_canvas_mean = tk.Canvas(self.labelframe_parameter, width=20, height=20)
        self.lasso_canvas_mean.grid(row=1, column=2)
        self.lasso_canvas_mean.create_oval(2,2,20,20, width=0, fill='#ff0000', tag='circle_mean')

        # 左手と右手のラベル
        label_left = tk.Label(self.labelframe_parameter, text='左手', font=("MS Pゴシック", 10, "bold"))
        label_left.grid(row=2, column=1)
        label_right = tk.Label(self.labelframe_parameter, text='右手', font=("MS Pゴシック", 10, "bold"))
        label_right.grid(row=2, column=3)

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

            label = tk.Label(self.labelframe_parameter, text=parameter_name[i], font=("MS Pゴシック", 10, "bold"))
            self.labels.append(label)
            label.grid(row=i+3, column=0, padx=10, pady=10)
            
            entry1 = tk.Entry(self.labelframe_parameter, textvariable=self.scale_values[i][0], width=3, state=tk.DISABLED)
            entry1.grid(row=i+3, column=2, padx=10)
            entry2 = tk.Entry(self.labelframe_parameter, textvariable=self.scale_values[i][1], width=3, state=tk.DISABLED)
            entry2.grid(row=i+3, column=4, padx=10)

            scale1 = tk.Scale(self.labelframe_parameter, to=180, variable=self.scale_values[i][0] ,orient=tk.HORIZONTAL, width=10, showvalue=False, length=150)
            scale1.grid(row=i+3, column=1, padx=10, pady=10)
            scale2 = tk.Scale(self.labelframe_parameter, to=180, variable=self.scale_values[i][1] ,orient=tk.HORIZONTAL, width=10, showvalue=False, length=150)
            scale2.grid(row=i+3, column=3, padx=10, pady=10)
            self.scales.append([scale1, scale2])
            
        
        #---------------------------------------
        # 動画再生
        #---------------------------------------
        self.labelframe_video = tk.LabelFrame(self.master, text="動画再生", width=810, height=620, font=("MS Pゴシック", 10, "bold"))
        self.labelframe_video.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_button = tk.Button(self.labelframe_video, command=self.push_play_button)
        self.video_button.pack(fill=tk.BOTH)
        self.labelframe_video.propagate(False)


    def push_file_analyze_button(self):
    
        self.filename = filedialog.askopenfilename(
            title = "ファイルの選択",
            # filetypes = [("PKL", ".pkl"), ("MP4", ".mp4"),("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = "./data/" # 自分自身のディレクトリ
        )

        try:
            _split = self.filename.split('/')
            self.folder_name = _split[-2]
            self.file_name = _split[-1].split('.')[0]
            ds = calculation.calculate_landmarks(self.filename)
            ds.landmarks = calculation.apply_moving_average(ds.landmarks)   # 移動平均を適用する
            ds.operation_time = calculation.calculate_operation_time(input_video_path=self.filename) # 動画の時間を計算する
            ds.joint_angles = calculation.calculate_joint_angle(ds.landmarks)                           # 関節の角度を計算する
            if ds.joint_angles == []:
                print(f'手を検出しなかった')
                return
            ds.joint_angle_mean = calculation.calculate_mean(ds.joint_angles)                           # 関節の角度の平均を計算する
            ds.joint_angle_var = calculation.calculate_variance(ds.joint_angles)                        # 関節の角度の分散を計算する
            # detection_state.save_detection_state(ds=ds, output_pkl_path=f'./data/{self.folder_name}/{self.file_name}.pkl')         # ランドマークを保存する
            with open(f'./data/{self.folder_name}/{self.file_name}.pkl', 'wb') as f:
                pickle.dump(self.ds, f)
            # ファイルの解析をした後、そのファイルを再生部分に表示する
            with open('GUI_settings.txt', 'w') as f:
                f.write(f'{self.folder_name} {self.file_name}')
            self.get_video(f'./data/{self.folder_name}/{self.file_name}.mp4')
            self.load_GUI_settings()
            self.load_detection_state()
            self.lasso_predict_sum = 0
            self.frame_cnt = 0
            self.combobox1.current(0 if self.gender=='man' else 1)
            self.combobox2.current(0 if self.dominant_hand=='Right' else 1)
            self.combobox3.current(0 if self.label=='1' else 1)
            self.realtime_flag = False
        
        except Exception as e:
            logger.error(e)

        


    def push_movie_button(self):
        self.sub_window = tk.Toplevel(self.master, width=250, height=250, )
        self.sub_window.title('動画撮影用ウィンドウ')
        self.sub_window.minsize(height=250, width=250) 
        
        self.start_button = tk.Button(self.sub_window, text='開始', command=self.push_start_movie_button, width=10, height=4)
        self.start_button.pack(side=tk.TOP, pady = 20)

        self.finish_button = tk.Button(self.sub_window, text='終了', command=self.push_finish_movie_button, width=10, height=4)
        self.finish_button.pack(side=tk.TOP, pady = 20)


    def push_start_movie_button(self):
        # リアルタイム処理でない場合は処理を行わない
        if self.realtime_flag == False and self.is_capture_started == True:
            return
        
        print("＜＜＜動画の撮影を開始しました＞＞＞")
        self.lasso_predict_sum = 0
        self.frame_cnt = 0
        self.start_button['state'] = tk.DISABLED
        self.is_capture_started = True
        
        # 動画保存用のvideowriterの設定
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS) #フレームレート取得
        fps = 15.0 # 30fpsだと動画が早送りになる
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #フォーマット指定
        
        self.save_movie_cnt = 1
        while os.path.isfile(f'./data/outputs/{self.save_movie_cnt}.mp4'):
            self.save_movie_cnt += 1
        self.writer = cv2.VideoWriter(f'./data/outputs/{self.save_movie_cnt}.mp4', fmt, fps, (width, height))

    
    def push_finish_movie_button(self):
        print("＞＞＞動画の撮影を終了しました＜＜＜")
        print(f'Lassoの予測値の平均値：{self.lasso_predict_sum/self.frame_cnt}')
        self.click_close_subwindow()
        self.is_capture_started = False
        self.start_button['state'] = tk.NORMAL
        try:
            self.writer.release()
        except Exception as e:
            pass
        
        # 撮影した動画を解析して動画再生部分に表示する
        self.lasso_mean_value = (self.lasso_predict_sum/self.frame_cnt)
        self.ds.lasso_mean_value = self.lasso_mean_value
        self.push_file_analyze_button()
        
        self.lasso_predict_sum = 0
        self.frame_cnt = 0


    
    def click_close_subwindow(self):
        try:
            self.writer.release()
        except Exception as e:
            pass
        self.sub_window = None


    def push_output_button(self):
        # リアルタイム処理をしているときは無効
        if self.realtime_flag:
            print('無効な操作です')
            return
        
        print(str(self.ds))
        
        
    def push_mediapipe_button(self):
        self.mediapipe_flag = not self.mediapipe_flag


    def push_realtime_button(self):
        self.realtime_flag = not self.realtime_flag
        self.video.release()
        
        self.lasso_predict_sum = 0.0
        self.frame_cnt = 0
        self.lasso_predict_mean.set(0.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        
        if self.realtime_flag:
            with open('GUI_settings.txt', 'w') as f:
                f.write(f'-1 -1')
            self.get_video()
            self.load_GUI_settings()
        else:
            self.open_filedialog()


    def click_close_mainwindow(self):
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
                self.gender             = self.ds.gender
                self.dominant_hand      = self.ds.dominant_hand
                self.label              = self.ds.label
                self.lasso_mean_value   = self.ds.lasso_mean_value
        except Exception as e:
            print('pklファイルの読み込みに失敗しました')
            logger.error(e)
        



    def open_filedialog(self):
        self.realtime_flag = False
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
        
        self.lasso_predict_sum = 0
        self.frame_cnt = 0
        self.combobox1.current(0 if self.gender=='man' else 1)
        self.combobox2.current(0 if self.dominant_hand=='Right' else 1)
        self.combobox3.current(0 if self.label=='1' else 1)


    def get_video(self, path=""):
        if path=="":
            self.video = cv2.VideoCapture(1)
        else:
            self.video = cv2.VideoCapture(path)


    def push_reload_button(self):
        if self.realtime_flag == True:
            print("無効な操作です")
            return
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
            ret, self.frame = self.video.read()
            self.lasso_predict_sum = 0
            self.frame_cnt = 0
        
        # 左利きにチェックボタンが入っていたら動画を反転させる．
        if self.is_left_handed.get()==True:
            self.frame = cv2.flip(self.frame,1)
        
        if self.is_capture_started == True and self.writer != None:
            self.writer.write(self.frame)

        if ret:
            tmp_image = copy.deepcopy(self.frame)
            if self.detector.detect(self.frame):
                tmp_image, tmp_landmark_dict = self.detector.draw(tmp_image)
                joint_angle = self.calculate_joint_angle_1frame(tmp_landmark_dict)

                for i in range(12):
                    if joint_angle['Left'] != []:
                        self.scale_values[i][0].set(int(joint_angle['Left'][i]))
                    if joint_angle['Right'] != []:
                        self.scale_values[i][1].set(int(joint_angle['Right'][i]))
                try:
                    x = []
                    x.extend(list(joint_angle['Left']))
                    x.extend(list(joint_angle['Right']))
                    pre = self.lasso.predict([x])[0]
                    pre = round(pre,2) # 小数点2桁へ丸める
                    self.realtime_lasso_predict.set(pre)
                    # Lassoの出力が閾値以上なら緑，以下なら赤色の円を描く
                    self.lasso_canvas_realtime.itemconfig('circle_realtime', fill='#00ff00' if pre > self.threshold else '#ff0000')
                    self.frame_cnt += 1
                    self.lasso_predict_sum += pre
                    self.lasso_predict_mean.set(round((self.lasso_predict_sum/self.frame_cnt), 2))
                    self.lasso_canvas_mean.itemconfig('circle_mean', fill='#00ff00' if self.lasso_predict_mean.get() > self.threshold else '#ff0000')
                except Exception as e:
                    pass
                

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