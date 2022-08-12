
import tkinter as tk            # ウィンドウ作成用
from tkinter import filedialog  # ファイルを開くダイアログ用
import numpy as np              # アフィン変換行列演算用
import os                       # ディレクトリ操作用
from PIL import Image, ImageTk  # 画像の処理
import cv2                      # 画像の処理
import subprocess               # サブプロセス
import pickle                   # オブジェクト形式での保存
from  pprint import pprint      # 表示用 
import time
import matplotlib.pyplot as plt

import calculation
import graphic

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.geometry("500x200+50+50")
        self.master.title("ウインドウ")
        #ウィジェット作成
        self.create_widgets()
        
    def create_widgets(self):
        self.menubar = tk.Menu(self)
        self.menu_file = tk.Menu(self.menubar, tearoff = False)
        self.menu_file.add_command(label = "ビデオの再生",  command = self.show_video)
        self.menu_file.add_command(label = "アニメーションの再生", command = self.show_animation)
        self.menu_file.add_command(label = "関節角度を表示", command = self.show_joint_angles)
        self.menubar.add_cascade(label="ファイル", menu = self.menu_file)
        self.master.config(menu = self.menubar)

        self.button = tk.Button(self.master, text="ファイル選択", width=10, command=self.show_video)
        self.button.pack()
    
    def open_filedialog(self):
        self.filename = filedialog.askopenfilename(
            title = "読み込むデータの選択",
            # filetypes = [("PKL", ".pkl"), ("MP4", ".mp4"),("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = "./data/9/" # 自分自身のディレクトリ
        )

    def show_video(self):
        self.open_filedialog()
        if self.filename == "": return
        
        # ランドマーク情報の計算を行う
        _ = calculation.calculate_landmarks(self.filename)
    
    def show_animation(self):
        self.open_filedialog()
        if self.filename == "": return

        # アニメーションを計算し再生する
        self.ds = calculation.calculate_landmarks(self.filename)
        gra = graphic.Graphic_3D(self.ds.landmarks)
        gra.plot()

    def show_joint_angles(self):
        self.open_filedialog()
        if self.filename == "": return

        graphic.show_joint_angles(pkl_path=self.filename)# ランドマークの時系列データを表示する
        

def main():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()