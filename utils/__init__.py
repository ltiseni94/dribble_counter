import cv2
import time
from typing import List, Any, Tuple


class FpsCounter:
    def __init__(self):
        """
        Build a FPS Counter
        """
        self.timer = cv2.getTickCount()

    def update(self) -> float:
        """
        Update the FPS counter and returns actual FPS

        :return:
            float: actual FPS
        """
        new_val = cv2.getTickCount()
        res = cv2.getTickFrequency() / (new_val - self.timer)
        self.timer = new_val
        return res


def log(msg: str) -> None:
    """
    Custom log function

    :param msg: msg to print
    :return: None
    """
    print(f'[{time.strftime("%H:%M:%S")}][Application]: {msg}')


def draw_bbox(img, bbox):
    """
    Draw a bounding box on a cv2 image.

    :param img: input image
    :param bbox: Bounding box in xywh format
    :return: modified image
    """
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
    return img


def calc_accuracy(pred: List[Any], true: List[Any]) -> float:
    """
    Calculate accuracy of predicted versus true values

    :param pred: List of predicted values
    :param true: List of true Values
    :return:
    """
    if len(pred) != len(true):
        raise ValueError('Mismatch length of predicted values')
    res = 0
    for idx, val in enumerate(pred):
        if val == true[idx]:
            res += 1
    return res / len(true)


def create_bounding_box(frame) -> Tuple[int, int, int, int]:
    """
    Open an interactive session in which the user can draw the start bounding box

    :param frame: input frame
    :return: bounding box in xywh format
    """
    orig_frame = frame.copy()
    cv2.namedWindow('first_frame', 1)
    rectangle_corners = []
    cnt = 0

    def mouse_callback(event, x, y, flags, param):
        """
        opencv Mouse Callback to attach to the image to capture user input
        """

        nonlocal frame, rectangle_corners, cnt
        if event == cv2.EVENT_LBUTTONDOWN and cnt < 2:
            rectangle_corners.append((x, y))
            cnt += 1
            log(f'point {cnt:1d}: ({x:4d}, {y:4d})')

        if event == cv2.EVENT_MOUSEMOVE and cnt == 1:
            frame = orig_frame.copy()
            cv2.rectangle(frame,
                          pt1=rectangle_corners[0],
                          pt2=(x, y),
                          color=(255, 0, 0),
                          thickness=2)

    cv2.setMouseCallback("first_frame", mouse_callback)

    while cnt < 2:
        cv2.imshow("first_frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    x_vals = (rectangle_corners[0][0], rectangle_corners[1][0])
    y_vals = (rectangle_corners[0][1], rectangle_corners[1][1])

    bbox = (min(x_vals), min(y_vals), abs(x_vals[1] - x_vals[0]), abs(y_vals[1] - y_vals[0]))
    log(f'Created Bounding Box: {bbox}')
    return bbox
