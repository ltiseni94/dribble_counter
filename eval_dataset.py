import os
import numpy as np
from typing import List, Tuple
from dribble_detector.metrics import compare_pred_true, compare_traj_true


def match_files(
        label_files: List[str],
        pred_files: List[str],
        traj_files: List[str],
) -> List[Tuple[str, str, str]]:
    results = []
    pred_files_copy = pred_files.copy()
    for label_file in label_files:
        for pred_file in pred_files_copy:
            if label_file.rstrip('_label.csv') == pred_file.rstrip('_pred.csv'):
                pred_files_copy.remove(pred_file)
                for traj_file in traj_files:
                    if traj_file.rstrip('_traj.csv') == label_file.rstrip('_label.csv'):
                        traj_files.remove(traj_file)
                        results.append((label_file, pred_file, traj_file))
                        break
                break
    return results


def main():
    label_files = [file for file in os.listdir('resources') if file.endswith('_label.csv')]
    pred_files = [file for file in os.listdir('resources') if file.endswith('_pred.csv')]
    traj_files = [file for file in os.listdir('resources') if file.endswith('_traj.csv')]
    matched_files = match_files(label_files, pred_files, traj_files)
    for label_file, pred_file, traj_file in matched_files:
        tp, fn, fp, acc = compare_pred_true(f'resources/{label_file}', f'resources/{pred_file}')
        traj_error = compare_traj_true(f'resources/{label_file}', f'resources/{traj_file}')
        print(
            f'Evaluated {label_file}:\t'
            f'Bounce TP: {tp:3d}\t'
            f'Bounce FP: {fp:3d}\t'
            f'Bounce FN: {fn:3d}\t'
            f'Accuracy: {(100*acc):4.1f}%\t\t'
            f'Trajectory error (avg): {np.mean(traj_error)}'
        )


if __name__ == '__main__':
    main()
