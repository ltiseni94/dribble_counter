import csv
import os
import sys
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


def main(dataset: str):
    if dataset[-1] == '/':
        dataset = dataset[:-1]
    label_files = [file for file in os.listdir(dataset) if file.endswith('_label.csv')]
    pred_files = [file for file in os.listdir(dataset) if file.endswith('_pred.csv')]
    traj_files = [file for file in os.listdir(dataset) if file.endswith('_traj.csv')]
    matched_files = match_files(label_files, pred_files, traj_files)
    try:
        os.mkdir(f'{dataset}/results')
    except FileExistsError:
        pass

    overall_results = []

    for label_file, pred_file, traj_file in matched_files:
        tp, fn, fp, acc, results = compare_pred_true(f'{dataset}/{label_file}', f'{dataset}/{pred_file}')
        traj_error = compare_traj_true(f'{dataset}/{label_file}', f'{dataset}/{traj_file}')
        print(
            f'Evaluated {label_file}:\t'
            f'Bounce TP: {tp:3d}\t'
            f'Bounce FP: {fp:3d}\t'
            f'Bounce FN: {fn:3d}\t'
            f'Accuracy: {(100*acc):4.1f}%\t\t'
            f'Trajectory error (avg): {np.mean(traj_error)}'
        )
        overall_results.append(dict(
            file=label_file[:-10],
            tp=tp,
            fp=fp,
            fn=fn,
            accuracy=round(100*acc, 2),
            traj_error=round(float(np.mean(traj_error)), 2),
            traj_len=len(traj_error),
        ))

        with open(f'{dataset}/results/{label_file[:-10]}_dribble.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['true', 'pred'])
            writer.writeheader()
            for res in results:
                writer.writerow({'true': res['label_data'], 'pred': res['pred_data']})

    with open(f'{dataset}/results/overall.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'tp', 'fp', 'fn', 'accuracy', 'traj_error', 'traj_len'])
        writer.writeheader()
        for res in overall_results:
            writer.writerow(res)


if __name__ == '__main__':
    main(sys.argv[1])
