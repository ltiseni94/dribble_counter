import cv2
import time
import mediapipe as mp
import csv
from typing import Union, Optional
from argparse import ArgumentParser
from dribble_detector.utils import FpsCounter, logger, draw_bbox, calc_accuracy, create_bounding_box, nice_text
from dribble_detector.hsv_tracker import HsvTracker
from dribble_detector.counter import ReboundCounter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def parse_args():
    parser = ArgumentParser(description="Dribble detector application. Open a video source (camera or a video file) "
                                        "Draw a bounding box centered on the ball with which the user want to do dribbles. "
                                        "Then the app will start counting dribbles. Using Pose tracking by MediaPipe, "
                                        "the app will detect the body segment with which the user perform the dribble")
    parser.add_argument('-t', '--tracker', action='store', default='CSRT',
                        help='Choose tracker type among: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW",'
                             ' "GOTURN", "MOSSE", "CSRT", "HVS"')
    parser.add_argument('-b', '--bbox', action='store', default=None, type=tuple,
                        help='Specify the starting bounding box for the ball'
                             ' through command line')
    parser.add_argument('-q', '--queue', '--queue-size', action='store',
                        default=5, type=int,
                        help='Queue size for average values calculation - filtering')
    parser.add_argument('-r', '--resize', '--resize-input', action='store',
                        default=1, type=int,
                        help='Decrease size of video input by specified scale factor')
    parser.add_argument('-s', '--source', action='store', default='resources/marcello.mp4',
                        help='Video source. Specify a path to a video or a camera')
    parser.add_argument('--bs', '--ball-start', action='store', default=None, type=int,
                        help='specify start index for average ball position value '
                             'when a dribble is detected')
    parser.add_argument('--be', '--ball-end', action='store', default=None, type=int,
                        help='specify end index for average ball position value '
                             'when a dribble is detected')
    parser.add_argument('--ps', '--pose-start', action='store', default=None, type=int,
                        help='specify start index for average pose position value '
                             'when a dribble is detected')
    parser.add_argument('--pe', '--pose-end', action='store', default=None, type=int,
                        help='specify end index for average pose position value '
                             'when a dribble is detected')
    parser.add_argument('--mindetection', '--min-detection-confidence',
                        action='store', default=0.5, type=float,
                        help='min detection confidence for mediapipe pose')
    parser.add_argument('--mintracking', '--min-tracking-confidence',
                        action='store', default=0.5, type=float,
                        help='min tracking confidence for mediapipe pose')
    parser.add_argument('--pause', '--pause-frame', action='store_true', default=False,
                        help='Stop frame for a second when a bounce is detected')
    parser.add_argument('--record-output', action='store_true', default=False,
                        help='Save output video in mp4 format')
    return parser.parse_args()


def select_tracker(tracker_type: str):
    if tracker_type == 'BOOSTING':
        t = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        t = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        t = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        t = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        t = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        t = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        t = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        t = cv2.TrackerCSRT_create()
    elif tracker_type == 'HSV':
        t = HsvTracker()
    else:
        logger.error(f'Could not find tracker {tracker_type}. Default to CSRT tracker')
        t = cv2.TrackerCSRT_create()
    return t


def main():
    args = parse_args()
    tracker = select_tracker(args.tracker)

    real_values_list = []

    source: Union[str, int] = args.source
    if not source.isnumeric():
        label_file = args.source.rstrip(args.source.split('.')[-1]) + 'csv'
        try:
            with open(label_file) as file:
                reader = csv.reader(file)
                for row in reader:
                    for val in row:
                        real_values_list.append(int(val))
        except ValueError:
            logger.error(f'File {label_file} contains bad label values')
        except FileNotFoundError:
            logger.warn(f'Could not find {label_file}')
        if args.source == 'resources/marcello.mp4':
            args.resize = 3
    else:
        source = int(source)

    video = cv2.VideoCapture(source)
    if not video.isOpened():
        logger.critical(f'Could not open video source {video}, exiting')
        exit(-1)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // args.resize
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) // args.resize
    frame_fps = int(video.get(cv2.CAP_PROP_FPS))
    logger.info(f'Cap size: ({width}, {height}); Cap fps: {frame_fps}')

    period = 1 / frame_fps
    # If the source is a camera, open it and collect the first ten frames
    # so that exposure is auto-adjusted by the camera software.
    if type(source) is int:
        for _ in range(10):
            _, _ = video.read()

    frame_read, frame = video.read()
    if not frame_read:
        logger.critical('Could not read frame, exiting')
        exit(-2)
    frame = cv2.resize(frame, (width, height))

    if args.bbox is None:
        bbox = create_bounding_box(frame, args.record_output)
    else:
        bbox = args.bbox
        logger.info(f'Got Input Bounding Box from command line: {bbox}')

    tracker.init(frame, bbox)

    reb = ReboundCounter(
        bbox,
        (width, height),
        ball_queue_indexes=(args.bs, args.be),
        pose_queue_indexes=(args.ps, args.pe),
    )

    predictions_list = []
    num_iter = 0
    pause_frame = False
    fps_counter = FpsCounter()

    video_writer: Optional[cv2.VideoWriter] = None
    if args.record_output:
        video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (width, height))

    with mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=args.mindetection,
            min_tracking_confidence=args.mintracking,
    ) as pose:
        while True:
            start_time = time.time()
            fps_counter.start()
            frame_read, frame = video.read()
            if not frame_read:
                if type(source) is int:
                    logger.critical('Camera error while reading frame')
                else:
                    logger.info('Processed all available frames')
                break
            num_iter += 1
            frame = cv2.resize(frame, (width, height))

            tracker_ok, bbox = tracker.update(frame)

            # Updating Pose detector. Set writeable False to speed up.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get FPS before the drawing tasks
            fps = fps_counter.update()

            if tracker_ok and results is not None:
                frame = draw_bbox(frame, bbox)
                is_bounce, label = reb.update(bbox, results)
                if is_bounce:
                    predictions_list.append(reb.label_converter[label])
                    if args.pause:
                        frame = nice_text(
                            frame,
                            f"{reb.label_dict[label]}",
                            position=(20, height - 50),
                        )
                        pause_frame = True
            else:
                frame = nice_text(
                    frame,
                    "Tracking failure detected",
                    position=(20, 80),
                    color=(60, 60, 255),
                )

            # Draw results
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            frame = nice_text(
                frame,
                f"{args.tracker} - FPS: {fps:.2f}",
                position=(80, 20),
            )

            # Print bounce counter
            frame = nice_text(
                frame,
                f"Bounces: {reb.get_total()}",
                position=(20, height-20),
            )
            # Print score counter
            frame = nice_text(
                frame,
                f"Score: {reb.get_score()}",
                position=(220, height - 20),
            )

            # Show image
            cv2.imshow("video", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            if args.record_output:
                video_writer.write(frame)
            if pause_frame:
                if args.record_output:
                    for _ in range(frame_fps - 1):
                        video_writer.write(frame)
                time.sleep(1.0)
                pause_frame = False
            elapsed_time = time.time() - start_time
            try:
                time.sleep(period - elapsed_time)
            except ValueError:
                pass

        if args.record_output:
            video_writer.release()
        video.release()

        logger.info(f'Run report:\n'
                    f'Processed {num_iter} frames\n'
                    f'Total bounces: {reb.get_total()}\n'
                    f'Average FPS: {sum(fps_counter.values)/len(fps_counter.values):.2f}')

        logger.info(f'{reb}\n')

        if real_values_list is not None:
            try:
                logger.info(f'Accuracy report:\n'
                            f'Labels    {real_values_list}\n'
                            f'Predicted {predictions_list}\n'
                            f'Accuracy: {calc_accuracy(predictions_list, real_values_list) * 100:.1f}%')
            except ValueError:
                logger.error(f'Could not calculate accuracy. Predictions list size is wrong. '
                             f'Check for false negatives or false positives.')


if __name__ == '__main__':
    main()
