# -*- coding:utf-8
from time import sleep, time
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from abst_detector import AbstDetector

from pprint import pprint

class HandTracker(AbstDetector):
    def __init__(self, max_num_hands: int, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        """初期化処理

        Args:
            max_num_hands (int): 最大検出手数
            min_detection_confidence (float): 手検出モデルの最小信頼値
            min_tracking_confidence (float): ランドマーク追跡モデルからの最小信頼値
        """
        self.tracker = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> bool:
        """手検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 手が検出できたか
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            self.results = self.tracker.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.multi_hand_landmarks is not None else False


    def draw(self, image: np.ndarray) -> tuple:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
            list      : ランドマーク
        """
        base_width, base_height = image.shape[1], image.shape[0]
        

        landmark_dict = {'Left':[], 'Right':[]}  # landmark_listをdict型で左手右手を取り出しやすいようにする
        landmark_color = {'Left':(255,0,0), 'Right':(0,255,0)}
        for i, (hand_landmarks, handedness) in enumerate(zip(self.results.multi_hand_landmarks, self.results.multi_handedness)):
            landmark_buf = []


            for landmark in hand_landmarks.landmark:
                landmark_buf.append([landmark.x, landmark.y, landmark.z])

                # 円を描く用の座標
                x = min(int(landmark.x * base_width), base_width - 1)
                y = min(int(landmark.y * base_height), base_height - 1)
                cv2.circle(image, (x, y), 4, landmark_color[str(handedness.classification[0].label)], 5)
            
            for con_pair in mp.solutions.hands.HAND_CONNECTIONS:
                # 節点の始点と終点の座標を計算する．
                u = (int(np.array(landmark_buf)[con_pair[0]][0]*base_width), int(np.array(landmark_buf)[con_pair[0]][1]*base_height))
                v = (int(np.array(landmark_buf)[con_pair[1]][0]*base_width), int(np.array(landmark_buf)[con_pair[1]][1]*base_height))
                cv2.line(image, u, v, landmark_color[str(handedness.classification[0].label)], 2)
            
            # ランドマークが欠損している場合は例外処理
            if len(landmark_buf) % 21 != 0:
                print("BREAK")
            
            landmark_dict[str(handedness.classification[0].label)] = landmark_buf


        return (image, landmark_dict)



            # print(landmark_buf)
            # print(handedness.classification[0].label)
            # print("---------------------------")
        # print(np.array(landmark_list).shape)
        # 辞書型で右手と左手のランドマークを保持する
        # for handness in self.results.multi_handedness:
        #     landmark_dict[str(handness.classification[0].label)] = landmark_list[handness.classification[0].index]
        
        # print(landmark_dict)

        # 上手くlandmark_dictに代入できていない！！！！！！！！！！！！！
        
            # for con_pair in mp.solutions.hands.HAND_CONNECTIONS:
            #     u = (int(np.array(landmark_buf)[con_pair[0]][0]*base_width), int(np.array(landmark_buf)[con_pair[0]][1]*base_height))
            #     v = (int(np.array(landmark_buf)[con_pair[1]][0]*base_width), int(np.array(landmark_buf)[con_pair[1]][1]*base_height))
            #     cv2.line(image, u, v, landmark_color[i], 2)
        
        # print(landmark_dict['Right'])

        # for i, hand in enumerate(['Left', 'Right']):
        #     for con_pair in mp.solutions.hands.HAND_CONNECTIONS:
        #         x1 = int(landmark_dict[hand][con_pair[i]][0]) * base_width
        #         y1 = int(landmark_dict[hand][con_pair[i]][1]) * base_height
        #         x2 = int(landmark_dict[hand][con_pair[i]][0]) * base_width
        #         y2 = int(landmark_dict[hand][con_pair[i]][1]) * base_height
        #         print(x1, y1, x2, y2)

        #         cv2.line(image, u, v, landmark_color[i], 2)   


        # 画像に円と線を描く．上のループに組み込みたい．
        # res_landmarks = []
        # for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
            
        #     # どっちの手か判別
        #     # print(handedness.classification[0].label)
        #     landmark_buf = []

        #     # keypoint
        #     for landmark in hand_landmarks.landmark:
        #         x = min(int(landmark.x * base_width), base_width - 1)
        #         y = min(int(landmark.y * base_height), base_height - 1)
        #         landmark_buf.append((x, y))
        #         res_landmarks.append([landmark.x, landmark.y, landmark.z])
        #         cv2.circle(image, (x, y), 4, (255, 0, 0), 5)
            
        #     # connection line
        #     for con_pair in mp.solutions.hands.HAND_CONNECTIONS:
        #         cv2.line(image, landmark_buf[con_pair[0]],
        #                 landmark_buf[con_pair[1]], (255, 0, 0), 2)
        # return (image, res_landmarks)