from typing import List, Optional, Tuple, Union
import csv
import cv2
import argparse
import os
from dribble_detector.utils import logger, draw_bbox, xyxy2xywh, nice_text
from dribble_detector.counter import ReboundCounter, make_label_csv_line


label_fields = ('frame', 'x1', 'y1', 'x2', 'y2', 'label')
QUIT_KEYS = tuple(map(ord, ('\x1b', 'q', 'Q')))
SAVE_KEYS = tuple(map(ord, ('\r', 's', 'S')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Video source for labeling. Can be a directory too')
    parser.add_argument('--read', '-r', action='store_true', default=False,
                        help='open source in read-only mode')
    parser.add_argument('--new-only', '-n', action='store_true', default=False,
                        help='perform labeling only on files without labels')
    return parser.parse_args()


def main(video: str, read: bool = False):
    if not video.endswith('.mp4'):
        raise ValueError(f'Wrong video input format "{video[-4:]}", should be ".mp4"')
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        res, frame = cap.read()
        if not res:
            if len(frames) == 0:
                logger.error(f'{video}: there are no frames available from this source')
                raise IOError
            logger.info(f'{video}: no more frames available. Read {len(frames)} frames')
            break
        frames.append(frame)

    frame_idx: int = 0
    temp_bbox: List[Tuple[int, int]] = []
    bounding_boxes: List[Optional[Tuple[Union[int, float], ...]]] = [None] * len(frames)
    dribble_labels: List[Optional[int]] = [None] * len(frames)
    try:
        with open(f'{video[:-4]}_label.csv', 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                idx = int(row['frame'])
                x1 = float(row['x1'])
                y1 = float(row['y1'])
                x2 = float(row['x2'])
                y2 = float(row['y2'])
                dribble_labels[idx] = int(row['label']) if row['label'] else None
                bounding_boxes[idx] = (x1, y1, x2, y2)
    except FileNotFoundError:
        logger.warning(f'Could not find label file: {video[:-4]}_label.csv')

    def draw_image():
        img = frames[frame_idx].copy()
        if bounding_boxes[frame_idx] is not None:
            img = draw_bbox(img, xyxy2xywh(bounding_boxes[frame_idx]))
        if dribble_labels[frame_idx] is not None:
            img = nice_text(img, ReboundCounter.label_from_numeric_value(dribble_labels[frame_idx]), (10, img.shape[0] - 10))
        cv2.imshow(f'{video}', img)

    def update_image(bbox):
        img = frames[frame_idx].copy()
        img = draw_bbox(img, xyxy2xywh(bbox))
        if dribble_labels[frame_idx] is not None:
            img = nice_text(img, ReboundCounter.label_from_numeric_value(dribble_labels[frame_idx]), (10, img.shape[0] - 10))
        cv2.imshow(f'{video}', img)

    def change_frame(frame_id: int):
        nonlocal frame_idx, temp_bbox
        temp_bbox = []
        frame_idx = frame_id
        cv2.setTrackbarPos('label', f'{video}', dribble_labels[frame_idx])
        draw_image()

    def change_label(label_id: int):
        nonlocal dribble_labels
        dribble_labels[frame_idx] = label_id
        draw_image()

    def mouse_callback(event, x, y, flags, params):
        nonlocal bounding_boxes, temp_bbox
        if event == cv2.EVENT_LBUTTONDOWN:
            if (flags & cv2.EVENT_FLAG_CTRLKEY) == cv2.EVENT_FLAG_CTRLKEY:
                if bounding_boxes[frame_idx] is not None:
                    logger.info(f'bounding box at frame {frame_idx} canceled')
                bounding_boxes[frame_idx] = None
                draw_image()
            elif (flags & cv2.EVENT_FLAG_SHIFTKEY) == cv2.EVENT_FLAG_SHIFTKEY:
                temp_bbox = temp_bbox[:-1]
                draw_image()
                logger.info(f'reset temp bounding box')
            else:
                if (pt := len(temp_bbox)) < 2:
                    temp_bbox.append((x, y))
                    logger.info(f'point {pt:1d}: ({x:4d}, {y:4d})')
                if len(temp_bbox) == 2:
                    bounding_boxes[frame_idx] = (*temp_bbox[0], *temp_bbox[1])
                    logger.info(f'Saved bbox {frame_idx}: {bounding_boxes[frame_idx]}')
                    temp_bbox.clear()
                    draw_image()

        if event == cv2.EVENT_MOUSEMOVE and len(temp_bbox) == 1:
            update_image((*temp_bbox[0], x, y))

    cv2.namedWindow(f'{video}')
    cv2.createTrackbar('frame', f'{video}', 0, len(frames) - 1, change_frame)
    cv2.createTrackbar('label', f'{video}', 0, len(ReboundCounter.label_converter) - 1, change_label)

    if not read:
        cv2.setMouseCallback(f'{video}', mouse_callback)
        print(
            f'\n\n'
            f'Instructions:\n'
            f'q, Q, esc: quit\n'
            f's, S, enter: save & quit (only if not in read mode)\n'
            f'mouse left click: add a point for a new bbox\n'
            f'mouse left click + shift: cancel previously added point\n'
            f'mouse left click + ctrl: cancel stored bbox\n'
        )

    draw_image()

    while True:
        if (key := (cv2.waitKey(10) & 0xFF)) in (*QUIT_KEYS, *SAVE_KEYS):
            break

    cv2.destroyAllWindows()

    if not read and key in SAVE_KEYS:
        with open(f'{video[:-4]}_label.csv', 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=label_fields)
            csv_writer.writeheader()
            bbox_to_write = ((idx, bbox) for idx, bbox in enumerate(bounding_boxes) if bbox is not None)
            for idx, bbox in bbox_to_write:
                new_row = make_label_csv_line(
                    frame=idx,
                    bbox=bbox,
                    label=dribble_labels[idx],
                )
                csv_writer.writerow(new_row)


if __name__ == '__main__':
    args = parse_args()
    source = args.source
    if os.path.isdir(source):
        if not source.endswith('/'):
            source += '/'
        videos = [source + video for video in os.listdir(source) if video.endswith('.mp4')]
        if args.new_only:
            labels = [source + label for label in os.listdir(source) if label.endswith('_label.csv')]
            for label in labels:
                try:
                    videos.remove(label.rstrip('_label.csv') + '.mp4')
                except ValueError:
                    pass
        for video in videos:
            main(video, args.read)
    else:
        if args.new_only:
            try:
                f = open(source[:-4] + '_label.csv', 'r')
                f.close()
            except FileNotFoundError:
                main(source, args.read)
        else:
            main(source, args.read)
