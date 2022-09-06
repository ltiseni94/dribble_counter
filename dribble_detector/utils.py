import cv2
import time
import logging
import numpy as np
from typing import List, Any, Tuple, Optional, Deque, Union
from collections import deque


class FpsCounter:
    def __init__(self):
        """
        Build an FPS Counter
        """
        self.timer: int = cv2.getTickCount()
        self.values: Deque[float] = deque([], maxlen=1000)

    def start(self) -> None:
        self.timer = cv2.getTickCount()

    def update(self) -> float:
        """
        Update the FPS counter and returns actual FPS

        :return:
            float: actual FPS
        """
        new_val = cv2.getTickCount()
        res = cv2.getTickFrequency() / (new_val - self.timer)
        self.values.append(res)
        return res


class ContextFilter(logging.Filter):
    logging_config = {
        'level': logging.INFO,
        'format': '%(opening)s[%(levelname)s] [%(asctime)s]: %(message)s%(closure)s',
        'datefmt': '%Y/%m/%d - %H:%M:%S',
    }
    reset_char = "\x1b[0m"
    color_dict = {
        int(logging.DEBUG): "\x1b[38m",
        int(logging.INFO): reset_char,
        int(logging.WARNING): "\x1b[33m",
        int(logging.ERROR): "\x1b[31m",
        int(logging.CRITICAL): "\x1b[31;1m",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        record.opening = self.color_dict[record.levelno]
        record.closure = self.reset_char
        return True


# Setup and create our custom logger with colored output
# and formatted date and time.
logging.basicConfig(**ContextFilter.logging_config)
logger = logging.getLogger()
logger.addFilter(ContextFilter())


def xyxy2xywh(bbox: Union[Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
    x_min = min([bbox[0], bbox[2]])
    x_max = max([bbox[0], bbox[2]])
    y_min = min([bbox[1], bbox[3]])
    y_max = max([bbox[1], bbox[3]])
    return x_min, y_min, x_max - x_min, y_max - y_min


def xywh2xyxy(bbox: Union[Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return x1, y1, x2, y2


def center_from_xyxy(bbox: Union[Tuple[int, ...], List[int]]) -> Tuple[int, int]:
    return (
        round((bbox[0] + bbox[2]) / 2),
        round((bbox[1] + bbox[3]) / 2),
    )


def center_from_xywh(bbox: Union[Tuple[int, ...], List[int]]) -> Tuple[int, int]:
    return (
        bbox[0] + round(bbox[2] / 2),
        bbox[1] + round(bbox[3] / 2),
    )


def draw_bbox(img: np.ndarray, bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...]]):
    """
    Draw a bounding box on a cv2 image.

    :param img: input image
    :param bbox: Bounding box in xywh format
    :return: modified image
    """
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (0, 0, 0), 4, 1)
    cv2.rectangle(img, p1, p2, (255, 255, 255), 2, 1)
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


def create_bounding_box(
        frame: np.ndarray,
        record_flag: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Open an interactive session in which the user can draw the start bounding box

    :param frame: input frame
    :param record_flag: If True, records a video of bounding box creation
    :return: bounding box in xywh format
    """
    window_name = "Draw bounding box"
    frame = nice_text(
        frame,
        'Use mouse left click',
        position=(10, frame.shape[0] - 40),
        size=0.6
    )
    frame = nice_text(
        frame,
        'Draw a rectangle around the ball',
        position=(10, frame.shape[0] - 15),
        size=0.6
    )
    orig_frame = frame.copy()
    cv2.namedWindow(window_name, 1)
    rectangle_corners = []
    cnt = 0

    video_writer: Optional[cv2.VideoWriter] = None
    if record_flag:
        video_writer = cv2.VideoWriter(
            'bounding.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            60,
            (frame.shape[1], frame.shape[0])
        )

    def mouse_callback(event, x, y, flags, param):
        """
        opencv Mouse Callback to attach to the image to capture user input
        """

        nonlocal frame, rectangle_corners, cnt
        if event == cv2.EVENT_LBUTTONDOWN and cnt < 2:
            rectangle_corners.append((x, y))
            cnt += 1
            logger.info(f'point {cnt:1d}: ({x:4d}, {y:4d})')

        if event == cv2.EVENT_MOUSEMOVE and cnt == 1:
            frame = orig_frame.copy()
            cv2.rectangle(frame,
                          pt1=rectangle_corners[0],
                          pt2=(x, y),
                          color=(0, 0, 0),
                          thickness=4)
            cv2.rectangle(frame,
                          pt1=rectangle_corners[0],
                          pt2=(x, y),
                          color=(255, 255, 255),
                          thickness=2)

    cv2.setMouseCallback(window_name, mouse_callback)

    while cnt < 2:
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        if record_flag:
            video_writer.write(frame)
        time.sleep(1 / 60)
    cv2.destroyAllWindows()

    x_vals = (rectangle_corners[0][0], rectangle_corners[1][0])
    y_vals = (rectangle_corners[0][1], rectangle_corners[1][1])

    bbox = (min(x_vals), min(y_vals), abs(x_vals[1] - x_vals[0]), abs(y_vals[1] - y_vals[0]))
    logger.info(f'Created Bounding Box: {bbox}')
    return bbox


def nice_text(
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        size: float = 0.75,
        thickness: int = 2,
) -> np.ndarray:

    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (0, 0, 0),
        thickness + 2
    )

    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness
    )

    return img


def nice_line(
        img: np.ndarray,
        line: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 4,
) -> np.ndarray:
    if thickness < 3:
        logger.warning(f'Thickness must be >= 3. Actual value is {thickness}')
        thickness = 3
    img = cv2.polylines(img, [np.array(line[:-1], dtype=np.int32).reshape((-1, 1, 2))], False, (0, 0, 0), thickness)
    img = cv2.polylines(img, [np.array(line[:-1], dtype=np.int32).reshape((-1, 1, 2))], False, color, thickness-2)
    return img


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Union[int, float]:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def array_distance(array1: np.ndarray, array2: np.ndarray):
    return np.array(((array1[0, :] - array2[0, :])**2 + (array1[1, :] - array2[1, :])**2) ** 0.5)