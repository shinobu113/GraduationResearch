from cgitb import text
from distutils import command
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


lock = th.Lock()

class VideoPlayer(tk.Frame):
    def __init__(self,master=None):
        super().__init__(master,width=1000,height=500)
        self.master.resizable(width=False, height=False)
        self.config(bg="#000000")
        # self.pack(expand=True,fill=tk.BOTH)
        self.video = None
        self.playing = False
        self.video_thread = None
        self.create_menu()
        # self.create_video_button()
        

    def create_menu(self):
        #---------------------------------------
        #  ツールバー
        #---------------------------------------
        self.frame_menubar = tk.Frame(self.master,relief = tk.SUNKEN, bd = 2)
        button1 = tk.Button(self.frame_menubar, text = "ファイルの選択", command=self.open_filedialog)
        button2 = tk.Button(self.frame_menubar, text = "項目の訂正")
        button3 = tk.Button(self.frame_menubar, text = "パラメータの調節")
        button4 = tk.Button(self.frame_menubar, text = "MediaPipe")
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
        combobox1 = ttk.Combobox(labelframe1, state="readonly", values=["男性","女性"])
        combobox1.pack(padx=10, pady=10)

        # 利き手
        labelframe2 = tk.LabelFrame(self.labelframe_parameter, text="利き手", height=40)
        labelframe2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        combobox2 = ttk.Combobox(labelframe2, state="readonly", values=["右手","左手"])
        combobox2.pack(padx=10, pady=10)

        # ラベル
        labelframe3 = tk.LabelFrame(self.labelframe_parameter, text="ラベル", height=40)
        labelframe3.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        combobox3 = ttk.Combobox(labelframe3, state="readonly", values=["1", "0"])
        combobox3.pack(padx=10, pady=10)

        #---------------------------------------
        # 動画再生
        #---------------------------------------
        self.labelframe_video = tk.LabelFrame(self.master, text="動画再生", width=810, height=620)
        self.labelframe_video.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_button = tk.Button(self.labelframe_video, command=self.push_play_button)
        self.video_button.pack(fill=tk.BOTH)
        self.labelframe_video.propagate(False)


    
    def open_filedialog(self):
        self.filename = filedialog.askopenfilename(
            title = "ファイルの選択",
            # filetypes = [("PKL", ".pkl"), ("MP4", ".mp4"),("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = "./data/" # 自分自身のディレクトリ
        )

    def get_video(self,path):
        self.video = cv2.VideoCapture(path)

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
            # messagebox.showerror("エラー","次のフレームがないので停止します")
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        else:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = self.video_button.winfo_width()/pil.width
            y = self.video_button.winfo_height()/pil.height
            ratio = x if x<y else y #三項演算子 xとyを比較して小さい方を代入
            # pil = pil.resize((int(ratio*pil.width),int(ratio*pil.height)))
            image = ImageTk.PhotoImage(pil)
            self.video_button.config(image=image)
            self.video_button.image = image
        lock.release()

    def video_frame_timer(self):
        while self.playing:
            self.next_frame()

if __name__ == "__main__":
    root = tk.Tk()
    path = "output.mp4"
    app = VideoPlayer(master=root)
    app.get_video(path)
    root.mainloop()