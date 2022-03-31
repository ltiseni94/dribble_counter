import time
from typing import NamedTuple, Tuple, Deque, Union, Dict, List, Optional
from collections import deque
from itertools import islice
from operator import itemgetter
import mediapipe as mp
import numpy as np


LandmarkEnum = mp.solutions.pose.PoseLandmark


class Rebounds(NamedTuple):
    ground: int = 0
    r_ft: int = 0
    l_ft: int = 0
    r_hip: int = 0
    l_hip: int = 0
    head: int = 0


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

    label_converter: Dict[str, int] = {
        'ground': -1,
        'r_ft': 0,
        'l_ft': 1,
        'r_hip': 2,
        'l_hip': 3,
        'head': 4,
    }

    label_dict: Dict[str, str] = {
        'ground': 'ground',
        'r_ft': 'right foot',
        'l_ft': 'left foot',
        'r_hip': 'right hip',
        'l_hip': 'left hip',
        'head': 'head',
    }

    landmark_dict: Dict[str, List[int]] = {
        'r_ft': [LandmarkEnum.RIGHT_FOOT_INDEX, LandmarkEnum.RIGHT_ANKLE],
        'l_ft': [LandmarkEnum.LEFT_FOOT_INDEX, LandmarkEnum.LEFT_ANKLE],
        'r_hip': [LandmarkEnum.RIGHT_KNEE, LandmarkEnum.RIGHT_HIP],
        'l_hip': [LandmarkEnum.LEFT_KNEE, LandmarkEnum.LEFT_HIP],
        'head': [LandmarkEnum.NOSE],
    }

    def __init__(self,
                 initial_bbox: Tuple[int, int, int, int],
                 img_shape: Tuple[int, int],
                 queue_size: int = 5,
                 *,
                 min_speed: Union[int, float] = 0.25,
                 ball_queue_indexes: Tuple[Optional[int], Optional[int]] = (None, None),
                 pose_queue_indexes: Tuple[Optional[int], Optional[int]] = (None, None),
                 initial_ball_queue: Optional[list] = None,
                 initial_mp_queue: Optional[list] = None,
                 lm_under_ball_threshold: int = -20,
                 ball_close_threshold: int = 80,):
        self._queue_size: int = queue_size
        self._min_speed: Union[int, float] = min_speed
        self._img_width: int = img_shape[0]
        self._img_height: int = img_shape[1]
        self.rebounds: Dict[str, int] = {}
        for label in self.label_converter:
            self.rebounds.update({label: 0})
        self._pos_queue: Deque[Tuple[Union[int, float], Union[int, float]]] = deque(
            [(int(initial_bbox[0] + initial_bbox[2]/2), int(initial_bbox[1] + initial_bbox[3]/2))] * queue_size,
            maxlen=queue_size
        )
        if initial_ball_queue is not None:
            self._pos_queue.clear()
            for item in initial_ball_queue:
                self._pos_queue.append(item)
        self._mp_queue: Deque[NamedTuple] = deque([], maxlen=queue_size)
        if initial_mp_queue is not None:
            self._mp_queue.clear()
            for item in initial_mp_queue:
                self._mp_queue.append(item)
        self._b_q_idxs: Tuple[Optional[int], Optional[int]] = ball_queue_indexes
        self._p_q_idxs: Tuple[Optional[int], Optional[int]] = pose_queue_indexes
        self._lm_under_ball_threshold: int = lm_under_ball_threshold
        self._ball_close_threshold: int = ball_close_threshold

    def __repr__(self):
        res = ''
        total = self.get_total() + self.rebounds['ground']
        for label, count in self.rebounds.items():
            tabs = '\t' * ((15 - len(self.label_dict[label])) // 8)
            res += f'\n{self.label_dict[label]}:\t{tabs}{count:03d}' \
                   f'\t{(self.rebounds[label]/total*100):03.1f}%'
        return f'Juggling summary:{res}'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def log(msg: str) -> None:
        print(f'[{time.strftime("%H:%M:%S")}][ReboundCounter]: {msg}')

    def get_total(self) -> int:
        return sum(self.rebounds.values()) - self.rebounds['ground']

    def _is_bounce(self,
                   prev: Union[int, float],
                   actual: Union[int, float]) -> bool:
        if prev == 0:
            prev = 0.25
        # self.log(f'{actual}, {prev}')
        return True if actual < 0 and prev*actual < -self._min_speed else False

    def update(self,
               bbox: Tuple[int, int, int, int],
               mp_results: NamedTuple) -> Tuple[bool, Optional[str]]:
        self._mp_queue.append(mp_results)
        previous_average_speed = sum(np.diff(self._pos_queue, axis=0))[1]/self._queue_size
        self._pos_queue.append((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)))
        actual_average_speed = sum(np.diff(self._pos_queue, axis=0))[1]/self._queue_size
        bounce = self._is_bounce(previous_average_speed, actual_average_speed)
        label = None
        if bounce:
            label = self._predict_rebound_label()
            self.rebounds[label] += 1
            if label != 'ground':
                self.log(f'Detected dribble with {self.label_dict[label]}')
            else:
                self.log(f'Ball fell on the ground')
        return bounce, label

    def _get_landmark_coord(self,
                            mp_results: NamedTuple,
                            *landmarks) -> Tuple[Union[int, float], Union[int, float]]:
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
        res_dict = {}
        for label, landmarks in self.landmark_dict.items():
            x = 0
            y = 0
            iter_list = list(islice(self._mp_queue, self._p_q_idxs[0], self._p_q_idxs[1]))
            assert len(iter_list) > 0
            for results in iter_list:
                lmx, lmy = self._get_landmark_coord(results, *landmarks)
                x += lmx
                y += lmy
            x /= len(iter_list)
            y /= len(iter_list)
            res_dict.update({label: (x, y)})
        return res_dict

    def _get_ball(self) -> Tuple[Union[int, float], Union[int, float]]:
        iter_list = list(islice(self._pos_queue, self._b_q_idxs[0], self._b_q_idxs[1]))
        res = np.sum(iter_list, axis=0)
        return res[0]/len(iter_list), res[1]/len(iter_list)

    @staticmethod
    def _calc_distance(p1: Tuple[Union[int, float], Union[int, float]],
                       p2: Tuple[Union[int, float], Union[int, float]]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _lm_is_valid(self,
                     lm_pos: Tuple[Union[int, float], Union[int, float]],
                     b_pos: Tuple[Union[int, float], Union[int, float]]) -> Tuple[float, bool]:
        under_ball = (lm_pos[1] - b_pos[1]) > self._lm_under_ball_threshold
        dist = self._calc_distance(lm_pos, b_pos)
        sufficiently_close = dist < self._ball_close_threshold
        # self.log(f'\n'
        #          f'\tb_pos: {b_pos}\n'
        #          f'\tlm_pos: {lm_pos}\n'
        #          f'\tunder_ball: {under_ball}\n'
        #          f'\tsuff_close: {sufficiently_close}\n'
        #          f'\tdistance: {dist}')
        return dist, under_ball and sufficiently_close

    def _predict_rebound_label(self):
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
        return 'ground'
