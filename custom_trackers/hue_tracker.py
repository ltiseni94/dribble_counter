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


    def update(self, image: np.ndarray):

        #idea: cerco hue medio della bbox e setto un'intervallo di colori
        #lo tengo come riferimento, allargo la bbox e creo una maschera con quel range di colori
        #trovo il centro della palla nella bbox allargata e restringo la bbox intorno alla palla

        #1: crop bbox
        crop_bbox = image[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2], :]
        #2: HUE
        bbox_hsv = cv2.cvtColor(crop_bbox, cv2.COLOR_BGR2HSV)
        self.mean_h = np.mean(bbox_hsv[:, :, 0])         #forse è meglio la moda
        self.min_h = self.mean_h - self.delta_h
        self.max_h = self.mean_h + self.delta_h

        #3: crop bbox allargata
        crop_bbox_new = image[self.bbox[1] - round(self.bbox[3]/2): self.bbox[1] + self.bbox[3] + round(self.bbox[3]/2),
               self.bbox[0] - round(self.bbox[2]/2): self.bbox[0] + self.bbox[2] + round(self.bbox[2]/2), :]

        #4: maschera su nuovo crop
        if self.min_h > self.max_h:
            mask = cv2.bitwise_not(cv2.inRange(crop_bbox_new[:, :, 0], self.max_h, self.min_h))
        else:
            mask = cv2.inRange(crop_bbox_new[:, :, 0], self.min_h, self.max_h)
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
