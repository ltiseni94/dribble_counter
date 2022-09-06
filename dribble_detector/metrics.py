import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Iterable, Optional, List, Dict
from .utils import distance, array_distance


def get_parabola_coefficients(
        pt0: Tuple[Union[float, int], ...],
        ptm: Tuple[Union[float, int], ...],
) -> Tuple[float, float, float]:
    """Returns parabola coefficient given a point and a maximum

    Formulas:
    y = a*t**2 + b*t + c
    :param pt0:
    :param ptm:
    :return:
    """

    if ptm[0] != 0:
        c = (pt0[2] * ptm[0] ** 2 - 2 * ptm[0] * pt0[0] * ptm[2] + ptm[2] * pt0[0] ** 2) / (pt0[0] - ptm[0]) ** 2
        a = (c - ptm[2]) / ptm[0] ** 2
        b = - 2 * a * ptm[0]
    else:
        c = ptm[2]
        b = 0
        a = (pt0[2] - ptm[2]) / pt0[0] ** 2
    return a, b, c


def parabola(
        t: Union[int, float, Iterable[Union[int, float]], np.ndarray],
        a: Union[int, float],
        b: Union[int, float],
        c: Union[int, float],
) -> Union[int, float, np.ndarray]:
    if type(t) in (tuple, list):
        t = np.array(t)
    return a * t ** 2 + b * t + c


def interpolate(
        pt0: Tuple[int, Union[float, int], Union[float, int], Optional[int]],
        pt1: Tuple[int, Union[float, int], Union[float, int], Optional[int]],
) -> np.ndarray:

    if pt0[2] > pt1[2]:
        ptm = pt1
    else:
        ptm = pt0
        pt0 = pt1

    label = pt0[-1]
    if label is None:
        raise ValueError("Missing label")

    num_frames = abs(ptm[0] - pt0[0])
    a, b, c = get_parabola_coefficients(pt0, ptm)
    t0 = min([ptm[0], pt0[0]])
    t = np.linspace(t0, t0 + num_frames - 1, num_frames).astype(np.int32)
    x = np.linspace(pt0[1], ptm[1], num_frames + 1).astype(np.int32)
    labels = np.zeros_like(t)
    if ptm[0] < pt0[0]:
        x = x[::-1]
    else:
        labels[0] = label
    x = x[:-1]
    y = parabola(t, a, b, c).astype(np.int32)
    return np.array([t, x, y, labels])


def get_point_from_label_file_row(row: Dict[str, str]) -> \
        Tuple[int, Union[float, int], Union[float, int], Optional[int]]:
    time = int(row['frame'])
    x = round((int(row['x1']) + int(row['x2'])) / 2)
    y = round((int(row['y1']) + int(row['y2'])) / 2)
    try:
        dribble_label = int(row['label'])
    except ValueError:
        dribble_label = None
    return time, x, y, dribble_label


def read_label_file(label_file: str) -> List[Tuple[int, Union[float, int], Union[float, int], Optional[int]]]:
    with open(label_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        return [get_point_from_label_file_row(row) for row in csv_reader]


def trajectory_from_csv(label_file: str, debug: bool = False) -> np.ndarray:
    points = read_label_file(label_file)
    trajectory = np.concatenate([interpolate(points[i], points[i + 1]) for i in range(len(points) - 1)], axis=1, dtype=np.int32)
    trajectory = np.append(trajectory, [[points[-1][0]], [points[-1][1]], [points[-1][2]], [points[-1][3]]], axis=1)
    if debug:
        plt.figure()
        plt.plot(trajectory[0, :], trajectory[2, :])
        plt.figure()
        plt.plot(trajectory[0, :], trajectory[1, :])
        plt.figure()
        plt.plot(trajectory[1, :], trajectory[2, :])
        plt.show()
    return trajectory


def compare_pred_true(label_file: str, pred_file: str):
    label = [lab for lab in read_label_file(label_file) if lab[-1] is not None]
    pred = read_label_file(pred_file)
    results = []
    pred_copy = pred.copy()
    for label_data in label:
        bounce_match = False
        label_match = False
        predicted_label = None
        for pred_data in pred_copy:
            if 0 <= (pred_data[0] - label_data[0]) < 5:
                if distance(pred_data[1:3], label_data[1:3]) < 80:
                    pred_copy.remove(pred_data)
                    bounce_match = True
                    label_match = label_data[-1] == pred_data[-1]
                    predicted_label = pred_data[-1]
        results.append(dict(
            frame=label_data[0],
            bounce=bounce_match,
            label_data=label_data[-1],
            pred_data=predicted_label,
            label_match=label_match)
        )
    bounce_true_positives = len(results)
    bounce_false_positives = len(pred_copy)
    bounce_false_negatives = len(label) - bounce_true_positives
    dribble_accuracy = len([res for res in results if res['label_match']]) / bounce_true_positives
    return bounce_true_positives, bounce_false_negatives, bounce_false_positives, dribble_accuracy, results


def compare_traj_true(label_file: str, traj_file: str):
    true_trajectory: np.ndarray = trajectory_from_csv(label_file)[0:3, :]
    with open(traj_file, 'r') as f:
        reader = csv.DictReader(f)
        t, x, y = [], [], []
        for row in reader:
            t.append(int(row['frame']))
            x.append(int(row['x']))
            y.append(int(row['y']))
    start_index = int(true_trajectory[0, 0])
    end_index = int(true_trajectory[0, -1])
    pred_trajectory = np.array([t, x, y], dtype=np.int32)
    pred_trajectory = pred_trajectory[:, (pred_trajectory[0, :] >= start_index) & (pred_trajectory[0, :] <= end_index)]
    if not (len(pred_trajectory) == len(true_trajectory)):
        raise ValueError('Trajectories have different length')
    return array_distance(true_trajectory[1:], pred_trajectory[1:])


if __name__ == '__main__':
    trajectory_from_csv('resources/marcello_label.csv', debug=True)
