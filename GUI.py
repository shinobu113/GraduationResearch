from cgitb import text
from distutils import command
from logging import exception
from textwrap import fill
from turtle import width
from cv2 import setIdentity
import numpy as np
import pickle
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

        self.load_GUI_settings()
        self.load_detection_state()
        self.create_menu()
        

    def create_menu(self):
        #---------------------------------------
        #  ツールバー
        #---------------------------------------
        self.frame_menubar = tk.Frame(self.master,relief = tk.SUNKEN, bd = 2)
        button1 = tk.Button(self.frame_menubar, text = "ファイルの選択", command=self.open_filedialog)
        button2 = tk.Button(self.frame_menubar, text = "項目の訂正")
        button3 = tk.Button(self.frame_menubar, text = "パラメータの調節")
        button4 = tk.Button(self.frame_menubar, text = "MediaPipe", command=self.push_mediapipe_button)
        # ボタンをフレームに配置
        button1.pack(side = tk.LEFT)
        button2.pack(side = tk.LEFT)
        button3.pack(side = tk.LEFT)
        button4.pack(side = tk.LEFT)
        # ツールバーをウィンドの上に配置
        self.frame_menubar.pack(side=tk.TOP, fill=tk.X)

        #---------------------------------------
        #  パラメータ調節
        #---------------------------------------
        self.labelframe_parameter = tk.LabelFrame(self.master, text="パラメータの調整", width=300, height=450)
        self.labelframe_parameter.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.labelframe_parameter.propagate(False)

        # 性別
        labelframe1 = tk.LabelFrame(self.labelframe_parameter, text="性別", height=40)
        labelframe1.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox1 = ttk.Combobox(labelframe1, state="readonly", values=["man","woman"])
        self.combobox1.pack(padx=10, pady=10)
        self.combobox1.current(0 if self.gender=='man' else 1)


        # 利き手
        labelframe2 = tk.LabelFrame(self.labelframe_parameter, text="利き手", height=40)
        labelframe2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox2 = ttk.Combobox(labelframe2, state="readonly", values=["Right","Left"])
        self.combobox2.pack(padx=10, pady=10)
        self.combobox2.current(0 if self.dominant_hand=='Right' else 1)

        # ラベル
        labelframe3 = tk.LabelFrame(self.labelframe_parameter, text="ラベル", height=40)
        labelframe3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.combobox3 = ttk.Combobox(labelframe3, state="readonly", values=["1", "0"])
        self.combobox3.pack(padx=10, pady=10)
        self.combobox3.current(0 if self.label==1 else 1)

        self.reload_button = tk.Button(self.labelframe_parameter, command=self.push_reload_button ,text="Reload", height=5, font=("",20), relief=tk.GROOVE)
        self.reload_button.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        #---------------------------------------
        # 動画再生
        #---------------------------------------
        self.labelframe_video = tk.LabelFrame(self.master, text="動画再生", width=810, height=620)
        self.labelframe_video.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_button = tk.Button(self.labelframe_video, command=self.push_play_button)
        self.video_button.pack(fill=tk.BOTH)
        self.labelframe_video.propagate(False)


    def destroy_widjet(self) -> None:
        self.frame_menubar.destroy()
        self.labelframe_parameter.destroy()
        self.labelframe_video.destroy()


    def push_mediapipe_button(self):
        self.mediapipe_flag = not self.mediapipe_flag


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
            # MediaPipeの検出器を作成
            self.detector = hand_tracker.HandTracker(
                2, 0.7, 0.5
            )
        except Exception as e:
            print('例外発生です')
            logger.error(e)


    def open_filedialog(self):
        self.filename = filedialog.askopenfilename(
            title = "ファイルの選択",
            # filetypes = [("PKL", ".pkl"), ("MP4", ".mp4"),("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = "./data/" # 自分自身のディレクトリ
        )
        _split = self.filename.split('/')
        self.folder_name = _split[-2]
        self.file_name = _split[-1].split('.')[0]
        
        with open('GUI_settings.txt', 'w') as f:
            f.write(f'{self.folder_name} {self.file_name}')
        
        self.destroy_widjet()
        self.load_GUI_settings()
        self.load_detection_state()
        self.create_menu()


    def get_video(self, path:str):
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

            rgb = cv2.cvtColor(tmp_image if self.mediapipe_flag else self.frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            image = ImageTk.PhotoImage(pil)

            self.video_button.config(image=image)
            self.video_button.image = image
        lock.release()


    def video_frame_timer(self):
        while self.playing:
            self.next_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(master=root)
    root.mainloop()