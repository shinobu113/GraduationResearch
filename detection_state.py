from codecs import getencoder
from re import A
import numpy as np
import hand_tracker
import cv2

class detection_state():
    detector            :hand_tracker.HandTracker
    cap                             = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    detected_hands_num  :int        = 0
    landmark            :np.ndarray = np.array([])
    gender              :bool
    dominant_hand       :str    #利き手('right' or 'left')←検出した手(handness)とは異なることに注意する．

    def __init__(self, detector, cap) -> None:
        self.detected_hands_num = 0
        self.detector = detector
        self.cap = cap
