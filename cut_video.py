import cv2
import csv
import argparse
from typing import Optional


QUIT_KEYS = tuple(map(ord, ('\x1b', 'q', 'Q')))
SAVE_KEYS = tuple(map(ord, ('\r', 's', 'S')))


def int_or_none(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, help='Video source for video cut')
    parser.add_argument('--output', '-o', type=str, default='', help='Video output file name, with extension')
    parser.add_argument('--convert-label', '-l', action='store_true', default=False)
    return parser.parse_args()


def main(video: str, output: str, convert_label: bool):
    frames = []
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Error: Could not open video')
        return
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while True:
            res, frame = cap.read()
            if not res:
                if len(frames) == 0:
                    print('Error: No frames available')
                    return
                print(f'No more frames available, {len(frames)} frames read')
                break
            frames.append(frame)
    finally:
        cap.release()

    start_frame: int = 0
    end_frame: int = len(frames) - 1

    def set_start_frame(frame_id: int):
        nonlocal start_frame
        start_frame = frame_id
        change_frame(frame_id)

    def set_end_frame(frame_id: int):
        nonlocal end_frame
        end_frame = frame_id
        change_frame(frame_id)

    def change_frame(frame_id: int):
        cv2.imshow(f'{video}', frames[frame_id])

    cv2.namedWindow(f'{video}')
    cv2.createTrackbar('start_frame', f'{video}', 0, len(frames) - 1, set_start_frame)
    cv2.createTrackbar('end_frame', f'{video}', len(frames) - 1, len(frames) - 1, set_end_frame)
    cv2.createTrackbar('frame', f'{video}', 0, len(frames) - 1, change_frame)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key in QUIT_KEYS:
            cv2.destroyAllWindows()
            return
        if key in SAVE_KEYS:
            cv2.destroyAllWindows()
            break

    if convert_label:
        try:
            with open(f'{video[:-4]}_label.csv', 'r') as f:
                labels = []
                for line in csv.DictReader(f):
                    labels.append({k: int_or_none(v) for k, v in line.items()})
        except FileNotFoundError:
            print('Error: label file not found')
            return

        for line in labels:
            line['frame'] -= start_frame

        try:
            with open(f'{video[:-4]}_label_cut.csv', 'x') as f:
                file_writer = csv.DictWriter(f, fieldnames=['frame', 'x1', 'y1', 'x2', 'y2', 'label'])
                file_writer.writeheader()
                for line in labels:
                    file_writer.writerow(line)
        except FileExistsError:
            print('Error: Attempting to create label file already existing')
            return

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    try:
        for frame in frames[start_frame: end_frame+1]:
            writer.write(frame)
    finally:
        writer.release()


if __name__ == '__main__':
    args = parse_args()
    if args.output == '':
        args.output = args.video[:-4] + '_cut.mp4'
    main(**vars(args))
