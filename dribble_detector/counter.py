from typing import NamedTuple, Tuple, Deque, Union, Dict, List, Optional
from collections import deque
from itertools import islice
from operator import itemgetter
import mediapipe as mp
import numpy as np
from .utils import logger


LandmarkEnum = mp.solutions.pose.PoseLandmark


def make_label_csv_line(
        frame: int,
        bbox: Union[List[int], Tuple[int, ...]],
        label: Optional[str] = None,
) -> Dict[str, int]:
    return dict(
        frame=frame,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        label=label,
    )


class Label:
    GROUND = 'ground'
    RIGHT_FOOT = 'right foot'
    LEFT_FOOT = 'left foot'
    RIGHT_HIP = 'right hip'
    LEFT_HIP = 'left hip'
    HEAD = 'head'

    def __iter__(self):
        return iter([self.GROUND, self.RIGHT_FOOT, self.RIGHT_HIP,
                     self.LEFT_FOOT, self.LEFT_HIP, self.HEAD])


class ReboundCounter:
    __slots__ = (
        '_queue_size',
        '_min_speed',
        '_img_width',
        '_img_height',
        '_b_q_idxs',
        '_p_q_idxs',
        '_lm_under_ball_threshold',
        '_ball_close_threshold',
        '_pos_queue',
        '_mp_queue',
        'rebounds',
    )

    label_converter: Dict[int, Optional[str]] = {
        0: None,
        1: Label.GROUND,
        2: Label.RIGHT_FOOT,
        3: Label.LEFT_FOOT,
        4: Label.RIGHT_HIP,
        5: Label.LEFT_HIP,
        6: Label.HEAD,
    }

    value_converter: Dict[Optional[str], int] = {
        v: k for k, v in label_converter.items()
    }

    label_scores: Dict[str, int] = {
        Label.GROUND: 0,
        Label.RIGHT_FOOT: 1,
        Label.LEFT_FOOT: 1,
        Label.RIGHT_HIP: 3,
        Label.LEFT_HIP: 3,
        Label.HEAD: 5,
    }

    landmark_dict: Dict[str, List[int]] = {
        Label.RIGHT_FOOT: [LandmarkEnum.RIGHT_FOOT_INDEX, LandmarkEnum.RIGHT_ANKLE],
        Label.LEFT_FOOT: [LandmarkEnum.LEFT_FOOT_INDEX, LandmarkEnum.LEFT_ANKLE],
        Label.RIGHT_HIP: [LandmarkEnum.RIGHT_KNEE, LandmarkEnum.RIGHT_HIP],
        Label.LEFT_HIP: [LandmarkEnum.LEFT_KNEE, LandmarkEnum.LEFT_HIP],
        Label.HEAD: [LandmarkEnum.NOSE],
    }

    @classmethod
    def label_from_numeric_value(cls, value: int) -> Optional[str]:
        return cls.label_converter[value]

    @classmethod
    def numeric_value_from_label(cls, label: str) -> int:
        if label not in Label():
            logger.warning(f'Label: {label} not recognized')
        return cls.value_converter.get(label, 0)

    def __init__(self,
                 initial_bbox: Tuple[int, int, int, int],
                 img_shape: Tuple[int, int],
                 *,
                 queue_size: int = 5,
                 min_speed: Union[int, float] = 0.25,
                 ball_queue_indexes: Tuple[Optional[int], Optional[int]] = (None, None),
                 pose_queue_indexes: Tuple[Optional[int], Optional[int]] = (None, None),
                 lm_under_ball_threshold: int = -20,
                 ball_close_threshold: int = 80, ):
        """
        :param initial_bbox:
            Initial bounding box in xywh format
        :param img_shape:
            Shape of the image coming from the video source (width, height)
        :param queue_size:
            Maximum number of samples stored for filtering ball position and human pose
        :param min_speed:
            Minimum speed required to detect a valid dribble
        :param ball_queue_indexes:
            (start_index, end_index) used for slicing data from the data queue when filtering ball speed
        :param pose_queue_indexes:
            (start_index, end_index) used for slicing data from the data queue when filtering
            pose tracking results - joint positions.
        :param lm_under_ball_threshold:
            Minimum vertical distance required - pixels - for a joint to be considered under the ball.
            Under the ball if positive, over the ball if negative.
        :param ball_close_threshold:
            Minimum distance required - pixels - for a joint to be considered sufficiently close to the ball.
            If none of the joint are close to the ball and a dribble has happened, we label it as "ground" dribble.
        """

        self._queue_size: int = queue_size
        self._min_speed: Union[int, float] = min_speed
        self._img_width: int = img_shape[0]
        self._img_height: int = img_shape[1]
        self.rebounds: Dict[str, int] = {label: 0 for label in self.label_scores}
        self._pos_queue: Deque[Tuple[Union[int, float], Union[int, float]]] = deque(
            [(int(initial_bbox[0] + initial_bbox[2]/2), int(initial_bbox[1] + initial_bbox[3]/2))] * queue_size,
            maxlen=queue_size,
        )
        self._mp_queue: Deque[NamedTuple] = deque([], maxlen=queue_size)
        self._b_q_idxs: Tuple[Optional[int], Optional[int]] = ball_queue_indexes
        self._p_q_idxs: Tuple[Optional[int], Optional[int]] = pose_queue_indexes
        self._lm_under_ball_threshold: int = lm_under_ball_threshold
        self._ball_close_threshold: int = ball_close_threshold

    def __repr__(self):
        res = ''
        total = self.get_total() + self.rebounds[Label.GROUND]
        for label, count in self.rebounds.items():
            tabs = '\t' * ((15 - len(label)) // 8)
            res += f'\n{label}:\t{tabs}{count:3d}' \
                   f'\t{(self.rebounds[label] / total * 100):4.1f}%'
        return f'Juggling summary:{res}'

    def __str__(self):
        return self.__repr__()

    def get_total(self) -> int:
        """
        Get total number of bounces

        :return:
            int Num of bounces
        """
        return sum(self.rebounds.values()) - self.rebounds[Label.GROUND]

    def get_score(self) -> int:
        """
        Calculates session score

        :return:
            int Session score
        """
        return sum([num * self.label_scores[label] for label, num in self.rebounds.items()])

    def _is_bounce(self,
                   prev: Union[int, float],
                   actual: Union[int, float]) -> bool:
        """
        Detect if a bounce has happened

        :param prev: previous ball vertical speed
        :param actual: actual ball vertical speed
        :return:
            True if a bounce is detected, False otherwise
        """
        if prev == 0:
            prev = self._min_speed
        return True if actual < 0 and prev*actual < -self._min_speed else False

    def update(self,
               bbox: Union[Tuple[int, int, int, int], List[int]],
               mp_results: NamedTuple) -> Tuple[bool, Optional[str]]:
        """
        Perform algorithm step update

        :param bbox: New bounding box from tracker algorithm
        :param mp_results: New results data structure from MediaPipe Pose
        :return: (bounce, label)
            bounce: True if a bounce is detected, False otherwise
            label: Dribble label if a bounce is detected, None otherwise
        """
        if len(bbox) != 4:
            raise ValueError(f'Input bbox {bbox} does not have expected size 4')

        self._mp_queue.append(mp_results)

        previous_average_speed = sum(np.diff(self._pos_queue, axis=0))[1] / self._queue_size
        self._pos_queue.append((int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)))
        actual_average_speed = sum(np.diff(self._pos_queue, axis=0))[1]/self._queue_size
        bounce = self._is_bounce(previous_average_speed, actual_average_speed)

        label = None
        if bounce:
            label = self._predict_rebound_label()
            self.rebounds[label] += 1
            if label != Label.GROUND:
                logger.info(f'Detected dribble with {label}')
            else:
                logger.info(f'Ball fell on the ground')
        return bounce, label

    def _get_landmark_coord(self,
                            mp_results: NamedTuple,
                            *landmarks) -> Tuple[Union[int, float], Union[int, float]]:
        """
        :param mp_results: mediapipe Pose result data structure.
        :param landmarks: landmarks to consider to calculate average position
        :return: landmark position (x, y) in the image.
        """
        landmarks_num = len(landmarks)
        x = 0
        y = 0
        for lm in landmarks:
            landmark = (mp_results.pose_landmarks.landmark[lm])
            x += landmark.x
            y += landmark.y
        return (
            x * self._img_width / landmarks_num,
            y * self._img_height / landmarks_num,
        )

    def _get_landmarks(self) -> Dict[str, Tuple[float, float]]:
        """
        For each landmark label used, performs average calculation over last
        N mediapipe results data structure - depending on configure slicing
        indexes and data queue length.

        :return:
            Dictionary containing filtered positions (x, y) for each landmark
            label used - see class attributes landmark_dict and label_dict
        """
        iter_list = list(islice(self._mp_queue, self._p_q_idxs[0], self._p_q_idxs[1]))
        if not len(iter_list) > 0:
            raise ValueError("Error while slicing data from mediapipe data queue")
        res_dict = {}
        for label, landmarks in self.landmark_dict.items():
            x = 0
            y = 0
            for results in iter_list:
                lmx, lmy = self._get_landmark_coord(results, *landmarks)
                x += lmx
                y += lmy
            x /= len(iter_list)
            y /= len(iter_list)
            res_dict.update({label: (x, y)})
        return res_dict

    def _get_ball(self) -> Tuple[Union[int, float], Union[int, float]]:
        """
        :return:
            Ball filtered position (x, y)
        """
        iter_list = list(islice(self._pos_queue, self._b_q_idxs[0], self._b_q_idxs[1]))
        if not len(iter_list) > 0:
            raise ValueError("Error with ball data slicing indexes")
        res = np.sum(iter_list, axis=0)
        return res[0] / len(iter_list), res[1] / len(iter_list)

    @staticmethod
    def _calc_distance(p1: Tuple[Union[int, float], Union[int, float]],
                       p2: Tuple[Union[int, float], Union[int, float]]) -> float:
        """
        Euclidean distance

        :param p1: First point (x, y)
        :param p2: Second point (x, y)
        :return: Distance
        """
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _lm_is_valid(self,
                     lm_pos: Tuple[Union[int, float], Union[int, float]],
                     b_pos: Tuple[Union[int, float], Union[int, float]]) -> Tuple[float, bool]:
        """
        :param lm_pos: Landmark position (x, y)
        :param b_pos: Ball position (x, y)
        :return:
            (distance, flag): distance between landmark and ball and flag that is True
            if the dribble is considered valid, False if it does not match required conditions
            of landmark being under the ball and ball being sufficiently close to the landmark.
        """
        under_ball = (lm_pos[1] - b_pos[1]) > self._lm_under_ball_threshold
        dist = self._calc_distance(lm_pos, b_pos)
        sufficiently_close = dist < self._ball_close_threshold
        return dist, under_ball and sufficiently_close

    def _predict_rebound_label(self):
        """
        Predict label associated to a rebound / dribble. Calculates distance and
        flag for each label, then finds minimum distance with True flag, and
        returns associated label.

        :return:
            Dribble label
        """
        ball_pos = self._get_ball()
        landmarks_pos_dict = self._get_landmarks()
        res_dict = {}
        for label, pos in landmarks_pos_dict.items():
            dist, flag = self._lm_is_valid(pos, ball_pos)
            if flag:
                res_dict.update({label: dist})
        if len(res_dict.keys()) > 0:
            res = sorted(res_dict.items(), key=itemgetter(1))
            return res[0][0]
        return Label.GROUND
