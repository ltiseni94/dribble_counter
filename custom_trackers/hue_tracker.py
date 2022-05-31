import cv2
import numpy as np
from typing import List, Optional


def nothing(x):
    pass


class HueTracker:
    def __init__(self, min_number_points: int = 15):
        self.bbox: Optional[List[int]] = None
        self.bbox_width: Optional[int] = None
        self.bbox_height: Optional[int] = None
        self.low_hsv: Optional[np.ndarray] = None
        self.high_hsv: Optional[np.ndarray] = None
        self.min_number_points: int = min_number_points
        self.kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    def init(self, image: np.ndarray, bounding_box: List[int]):
        self.bbox = bounding_box
        self.bbox_width = bounding_box[2]
        self.bbox_height = bounding_box[3]
        #cv2.imshow('original frame', image)
        #cv2.waitKey(1)
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.namedWindow('mask')
        cv2.createTrackbar('low_h', 'mask', 0, 179, nothing)
        cv2.createTrackbar('high_h', 'mask', 179, 179, nothing)
        cv2.createTrackbar('low_s', 'mask', 0, 255, nothing)
        cv2.createTrackbar('high_s', 'mask', 255, 255, nothing)
        cv2.createTrackbar('low_v', 'mask', 0, 255, nothing)
        cv2.createTrackbar('high_v', 'mask', 255, 255, nothing)
        low_hsv = np.array([0, 0, 0], dtype=np.uint8)
        high_hsv = np.array([179, 255, 255], dtype=np.uint8)
        while True:
            low_hsv[0] = cv2.getTrackbarPos('low_h', 'mask')
            high_hsv[0] = cv2.getTrackbarPos('high_h', 'mask')
            low_hsv[1] = cv2.getTrackbarPos('low_s', 'mask')
            high_hsv[1] = cv2.getTrackbarPos('high_s', 'mask')
            low_hsv[2] = cv2.getTrackbarPos('low_v', 'mask')
            high_hsv[2] = cv2.getTrackbarPos('high_v', 'mask')
            self.set_hsv_mask(
                low_hsv,
                high_hsv
            )
            mask = self.create_mask(hsv_frame)
            image_mask = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow('mask', mask)
            cv2.imshow('frame AND mask', image_mask)
            k = cv2.waitKey(1)
            if k & 0xFF in (ord('q'), 27, ord('\n'), ord('\r')):
                break
        cv2.destroyAllWindows()

    def set_hsv_mask(self, low_hsv: np.ndarray, high_hsv: np.ndarray) -> None:
        self.low_hsv = low_hsv
        self.high_hsv = high_hsv

    def create_mask(self, frame: np.ndarray):
        if self.low_hsv[0] > self.high_hsv[0]:
            dummy_low_hsv = self.low_hsv.copy()
            dummy_low_hsv[0] = 0
            dummy_high_hsv = self.high_hsv.copy()
            dummy_high_hsv[0] = 179
            mask = cv2.bitwise_or(cv2.inRange(frame, dummy_low_hsv, self.high_hsv), cv2.inRange(frame, self.low_hsv, dummy_high_hsv))
        else:
            mask = cv2.inRange(frame, self.low_hsv, self.high_hsv)
        return cv2.erode(mask, self.kernel)

    def update(self, image: np.ndarray):
        width_bbox = self.bbox[2]
        heigth_bbox = self.bbox[3]
        delta_width = round(self.bbox[2]/2)
        delta_height = round(self.bbox[3]/2)

        if delta_height < self.bbox[1] and delta_width < self.bbox[0]:
            crop_bbox_new = image[self.bbox[1] - delta_height: self.bbox[1] + heigth_bbox + delta_height,
               self.bbox[0] - delta_width: self.bbox[0] + width_bbox + delta_width, :]
        else:
            crop_bbox_new = image[self.bbox[1]: self.bbox[1] + heigth_bbox + delta_height,
                            self.bbox[0]: self.bbox[0] + width_bbox + delta_width, :]
        if crop_bbox_new.size == 0:
            crop_bbox_new = image[self.bbox[1]:self.bbox[1]+self.bbox[3], self.bbox[0]:self.bbox[0]+self.bbox[2], :]

        bbox_new_hsv = cv2.cvtColor(crop_bbox_new, cv2.COLOR_BGR2HSV)
        mask = self.create_mask(bbox_new_hsv)

        ball_indexes = np.nonzero(mask)
        if len(ball_indexes[0]) < self.min_number_points:
            return False, None

        x_ball_crop, y_ball_crop = np.median(ball_indexes[1]).item(), np.median(ball_indexes[0]).item()

        x_ball_image = x_ball_crop + self.bbox[0] - round(self.bbox[2]/2)
        y_ball_image = y_ball_crop + self.bbox[1] - round(self.bbox[3]/2)

        self.bbox = [
            int(x_ball_image - self.bbox_width/2),
            int(y_ball_image - self.bbox_height/2),
            self.bbox_width,
            self.bbox_height
        ]
        return True, self.bbox
