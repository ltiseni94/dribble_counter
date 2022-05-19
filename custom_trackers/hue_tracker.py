import cv2
import numpy as np
from typing import List, Optional


class HueTracker:
    def __init__(self, mean_h: int = 180, delta_h: int = 20, min_number_points: int = 15):
        self.bbox: Optional[List[int]] = None
        self.bbox_width: Optional[int] = None
        self.bbox_height: Optional[int] = None
        self.mean_h: int = mean_h
        self.delta_h: int = delta_h
        self.min_h: int = mean_h - delta_h
        self.max_h: int = mean_h + delta_h
        self.min_number_points: int = min_number_points
        self.kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    def init(self, image: np.ndarray, bounding_box: List[int]):
        self.bbox = bounding_box
        self.bbox_width = bounding_box[2]
        self.bbox_height = bounding_box[3]

        #1: crop bbox
        crop_bbox = image[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2], :]
        #2: HUE
        bbox_hsv = cv2.cvtColor(crop_bbox, cv2.COLOR_BGR2HSV)
        self.mean_h = np.mean(bbox_hsv[:, :, 0])         #forse è meglio la moda
        self.min_h = self.mean_h - self.delta_h
        self.max_h = self.mean_h + self.delta_h


    def update(self, image: np.ndarray):
        #3: crop bbox allargata
        crop_bbox_new = image[self.bbox[1] - round(self.bbox[3]/2): self.bbox[1] + self.bbox[3] + round(self.bbox[3]/2),
               self.bbox[0] - round(self.bbox[2]/2): self.bbox[0] + self.bbox[2] + round(self.bbox[2]/2), :]
        if crop_bbox_new.size == 0:
            crop_bbox_new = image[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2], :]

        #4: maschera su nuovo crop
        bbox_new_hsv = cv2.cvtColor(crop_bbox_new, cv2.COLOR_BGR2HSV)
        if self.min_h > self.max_h:
            mask = cv2.bitwise_not(cv2.inRange(bbox_new_hsv[:, :, 0], self.max_h/2, self.min_h/2))
        else:
            mask = cv2.inRange(bbox_new_hsv[:, :, 0], self.min_h/2, self.max_h/2)
        mask = cv2.erode(mask, self.kernel)

        #5: centro palla
        ball_indexes = np.nonzero(mask)
        if len(ball_indexes[0]) < self.min_number_points:
            return False, None

        x_ball, y_ball = np.median(ball_indexes[1]).item(), np.median(ball_indexes[0]).item()

        #6: bbox aggiornata
        self.bbox = [
            int(x_ball - self.bbox_width/2),
            int(y_ball - self.bbox_height/2),
            self.bbox_width,
            self.bbox_height
        ]
        return True, self.bbox
