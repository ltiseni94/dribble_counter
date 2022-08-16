import cv2
import argparse
import time
import numpy as np
from dribble_detector.metrics import trajectory_from_csv
from dribble_detector.utils import nice_text, nice_line
from dribble_detector.counter import ReboundCounter
from collections import deque
from typing import Optional, Deque


LABEL_ONSCREEN_DURATION_FRAMES = 10


class DrawTrajectory:
    def __init__(self, maxlen: int = 15):
        self.should_remove: bool = False
        self.deque: deque = deque([], maxlen=maxlen)

    def update(self, point: Optional[np.ndarray] = None) -> bool:
        if self.should_remove:
            if len(self.deque) > 1:
                self.deque.popleft()
            else:
                return False
        else:
            if point is None:
                self.should_remove = True
            else:
                self.deque.append(point)
        return True

    def get_points(self) -> np.ndarray:
        return np.asarray(self.deque, dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='resources/marcello.mp4')
    parser.add_argument('--fps', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    source: str = args.source
    period: float = 1 / args.fps
    label_file = source.rstrip('.mp4') + '_label.csv'
    trajectory = trajectory_from_csv(label_file)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f'Could not open source: "{source}"')
    draw_trajectories: Deque[DrawTrajectory] = deque([])
    idx = 0
    label = None
    new_label_time = 0
    while True:
        start_time = time.time()
        res, frame = cap.read()
        if not res:
            break
        point = None
        if idx in trajectory[0, :]:
            point = trajectory[:, trajectory[0, :] == idx].squeeze()
            if point[-1] is not None and point[-1] != 0:
                label = point[-1]
                new_label_time = idx
            point = point[1:-1]
            if len(point) != 2:
                raise ValueError(f'new point: "{point}" does not match format')
            if all(map(lambda x: x.should_remove, draw_trajectories)):
                draw_trajectories.append(DrawTrajectory())
        res = [draw.update(point) for draw in draw_trajectories]
        for val in res:
            if not val:
                draw_trajectories.popleft()
        lines = list(map(lambda x: x.get_points(), draw_trajectories))
        for line in lines:
            if len(line) > 2:
                frame = nice_line(frame, line)
        if (idx - new_label_time) < LABEL_ONSCREEN_DURATION_FRAMES and label is not None:
            frame = nice_text(frame, f'{ReboundCounter.label_from_numeric_value(label)}', (10, frame.shape[0] - 10))
        else:
            label = None
        cv2.imshow(f'{source}', frame)
        elapsed_time = time.time() - start_time
        idx += 1
        if (diff := (period - elapsed_time)) > 0:
            key = cv2.waitKey(int(diff * 1000)) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            while True:
                key = cv2.waitKey(100)
                if key == ord(' '):
                    break


if __name__ == '__main__':
    main()
