import cv2
import numpy as np
from typing import List, Optional


class HueTracker:
    def __init__(self, min_h: int = 70, max_h: int = 80, min_number_points: int = 15):
        self.bbox: Optional[List[int]] = None
        self.bbox_width: Optional[int] = None
        self.bbox_height: Optional[int] = None
        self.min_h: int = min_h
        self.max_h: int = max_h
        self.min_number_points: int = min_number_points
        self.kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    def init(self, image: np.ndarray, bounding_box: List[int]):
        self.bbox = bounding_box
        self.bbox_width = bounding_box[2]
        self.bbox_height = bounding_box[3]

    def update(self, image: np.ndarray):
        frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.min_h > self.max_h:
            mask = cv2.bitwise_not(cv2.inRange(frame_hsv[:, :, 0], self.max_h, self.min_h))
        else:
            mask = cv2.inRange(frame_hsv[:, :, 0], self.min_h, self.max_h)
        mask = cv2.erode(mask, self.kernel)
        ball_indexes = np.nonzero(mask)

        if len(ball_indexes[0]) < self.min_number_points:
            return False, None

        x_ball, y_ball = np.median(ball_indexes[1]).item(), np.median(ball_indexes[0]).item()
        self.bbox = [
            int(x_ball - self.bbox_width/2),
            int(y_ball - self.bbox_height/2),
            self.bbox_width,
            self.bbox_height
        ]
        return True, self.bbox
