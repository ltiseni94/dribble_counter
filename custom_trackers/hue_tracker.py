import cv2
import numpy as np
from typing import List, Optional


class HueTracker:
    def __init__(self, mean_h: int = 90, delta: int = 20, min_number_points: int = 15):
        self.bbox: Optional[List[int]] = None
        self.bbox_width: Optional[int] = None
        self.bbox_height: Optional[int] = None
        self.mean_h: int = mean_h
        self.delta_h: int = delta
        self.low_h: int = mean_h - delta
        self.high_h: int = mean_h + delta
        self.lower_hsv: np.ndarray = np.array([0, 0, 0])
        self.higher_hsv: np.ndarray = np.array([0, 0, 0])
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
        self.mean_h = np.mean(bbox_hsv[:, :, 0])
        self.low_h = (self.mean_h - self.delta_h)/2
        self.high_h = (self.mean_h + self.delta_h)/2

        #2bis: scala HSV
        ilowH = self.low_h
        ihighH = self.high_h

        ilowS = np.mean(bbox_hsv[:, :, 1]) - 20
        ihighS = 255

        ilowV = np.mean(bbox_hsv[:, :, 2]) - 20
        ihighV = np.mean(bbox_hsv[:, :, 2]) + 20
        self.lower_hsv = np.array([ilowH, ilowS, ilowV])
        self.higher_hsv = np.array([ihighH, ihighS, ihighV])


    def update(self, image: np.ndarray):
        #3: crop bbox allargata
        crop_bbox_new = image[self.bbox[1] - round(self.bbox[3]/2): self.bbox[1] + self.bbox[3] + round(self.bbox[3]/2),
               self.bbox[0] - round(self.bbox[2]/2): self.bbox[0] + self.bbox[2] + round(self.bbox[2]/2), :]

        if crop_bbox_new.size == 0:
            crop_bbox_new = image[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2], :]

        #4: maschera su nuovo crop
        bbox_new_hsv = cv2.cvtColor(crop_bbox_new, cv2.COLOR_BGR2HSV)
        #if self.low_h > self.high_h:
            #mask = cv2.bitwise_not(cv2.inRange(bbox_new_hsv[:, :, 0], self.high_h, self.low_h))
        #else:
            #mask = cv2.inRange(bbox_new_hsv[:, :, 0], self.low_h, self.high_h)

        mask = cv2.inRange(bbox_new_hsv, self.lower_hsv, self.higher_hsv)

        mask = cv2.erode(mask, self.kernel)

        #5: centro palla
        ball_indexes = np.nonzero(mask)
        if len(ball_indexes[0]) < self.min_number_points:
            return False, None

        x_ball_crop, y_ball_crop = np.median(ball_indexes[1]).item(), np.median(ball_indexes[0]).item()

        x_ball_image = x_ball_crop + self.bbox[0] - round(self.bbox[2]/2)
        y_ball_image = y_ball_crop + self.bbox[1] - round(self.bbox[3]/2)

        #possibile raggio palla
        #r_ball = (np.max(ball_indexes[1]).item() - np.min(ball_indexes[1]).item())/2

        #6: bbox aggiornata
        self.bbox = [
            int(x_ball_image - self.bbox_width/2),
            int(y_ball_image - self.bbox_height/2),
            self.bbox_width,
            self.bbox_height
        ]
        return True, self.bbox
