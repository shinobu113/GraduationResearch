import numpy as np
import hand_tracker
import cv2

class detection_state():
    detector            :hand_tracker.HandTracker
    cap                             = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    detected_hands_num  :int        = 0
    landmark            :np.ndarray = np.array([])
    

    def __init__(self, detector, cap) -> None:
        self.detected_hands_num = 0
        self.detector = detector
        self.cap = cap
