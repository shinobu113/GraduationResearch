from time import sleep, time
import cv2
import numpy as np
import copy

import detection_state


def get_datas(ds: detection_state.detection_state) -> None:
    
    while ds.cap.isOpened():
        # 静止画またはカメラ入力
        ret, image = ds.cap.read()
        if not ret:
            break


        tmp_image = copy.deepcopy(image)
        tmp_image = cv2.circle(tmp_image, (35, 35), 20, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        tmp_landmark = []
        
        if ds.detector.detect(image):
            tmp_image, tmp_landmark = ds.detector.draw(tmp_image)
            # 手を検知して，欠損がないことを確認して手の検知数とランドマークを更新する．
            if (len(tmp_landmark)!=0 and len(tmp_landmark)%21 == 0):
                """ここでデータの収集と保存を行う．"""
                
                ds.landmark = np.array(tmp_landmark)
                ds.detected_hands_num = int(len(tmp_landmark)/21)
        
        cv2.imshow("hand_tracker", tmp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            break
    
    print("Collected Datas")
    