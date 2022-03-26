import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from argparse import ArgumentParser

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


idx_dict = [
    'right foot',
    'left foot',
    'right hip',
    'left hip',
    'right shoulder',
    'left shoulder',
    'head'
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}]: {msg}')


def is_bounce(prev, actual):
    if prev == 0:
        prev = 0.25
    return True if actual < 0 and prev*actual < -5 else False


def bounce_count(q: deque, box):
    prev_avg = sum(np.diff(q))
    q.append(box[1] + box[3]/2)
    new_avg = sum(np.diff(q))
    # log(f'{new_avg} - {prev_avg} - {is_bounce(prev_avg, new_avg)}')
    return is_bounce(prev_avg, new_avg)


def landmark_vector(landmark, w: int = 360, h: int = 640):
    return [landmark.x * w, landmark.y * h]


def distance_from_landmark(landmark, b_center):
    flag = landmark[1] - b_center[1] > -15
    return ((landmark[0] - b_center[0])**2 + (landmark[1] - b_center[1])**2)**0.5, flag


def distance_from_segment(landmark_1, landmark_2, b_center):
    landmark = [(landmark_2[0] + landmark_1[0])/2, (landmark_2[1] + landmark_1[1])/2]
    return distance_from_landmark(landmark, b_center)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--tracker', action='store', default='CSRT',
                        help='Choose tracker type among: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW",'
                        ' "GOTURN", "MOSSE", "CSRT"')

    arguments_ns = parser.parse_args()

    tracker_types = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT')
    tracker_type = arguments_ns.tracker

    if tracker_type not in tracker_types:
        log(f'Requested tracker {arguments_ns.tracker} is not available. Will default to "CSRT"')
        tracker_type = "CSRT"
    
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerMIL_create()

    real_values_list = [0, 1, 0, 1, 0, 3, 0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 3, 2, 3, 0, 0, 1, 0, 0]
    video = cv2.VideoCapture('./resources/marcello.mp4')
    if not video.isOpened():
        log('Could not open video, exiting')
        exit(-1)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//3
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//3
    log(f'Cap size: ({width}, {height})')

    frame_read, frame = video.read()
    if not frame_read:
        log('Could not read frame, exiting')
        exit(-2)
    frame = cv2.resize(frame, (width, height))

    cv2.namedWindow('first_frame', 1)

    orig_frame = frame.copy()
    rectangle_corners = []
    cnt = 0

    def mouse_callback(event, x, y, flags, param):
        global frame, rectangle_corners, cnt
        if event == cv2.EVENT_LBUTTONDOWN and cnt < 2:
            rectangle_corners.append((x, y))
            cnt += 1
            log(f'cnt = {cnt:1d}\t'
                f'x = {x:4d}\t'
                f'y = {y:4d}')
        if event == cv2.EVENT_MOUSEMOVE and cnt == 1:
            frame = orig_frame.copy()
            cv2.rectangle(frame,
                          pt1=rectangle_corners[0],
                          pt2=(x, y),
                          color=(255, 0, 0),
                          thickness=2)

    cv2.setMouseCallback("first_frame", mouse_callback)

    while cnt < 2:
        cv2.imshow("first_frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

    x_vals = (rectangle_corners[0][0], rectangle_corners[1][0])
    y_vals = (rectangle_corners[0][1], rectangle_corners[1][1])

    bbox = (min(x_vals), min(y_vals), abs(x_vals[1]-x_vals[0]), abs(y_vals[1]-y_vals[0]))
    log(f'Bounding Box: {bbox}')

    tracker_ok = tracker.init(orig_frame, bbox)

    bounce = 0
    predictions_list = []
    speed_queue = deque([bbox[1] + bbox[3]/2] * 5, 5)
    box_center_deque = deque([[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]] * 5, 5)
    mp_results_deque = deque([], 5)

    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        num_iter = 0
        pause_frame = False
        while True:
            frame_read, frame = video.read()
            if not frame_read:
                log('No more frames available')
                break
            num_iter += 1

            frame = cv2.resize(frame, (width, height))

            # get start calculation time
            timer = cv2.getTickCount()

            # Tracking ball
            tracker_ok, bbox = tracker.update(frame)

            # Updating Pose detector
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            mp_results_deque.append(results)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if tracker_ok:
                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                box_center = [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]
                box_center_deque.append(box_center)

                # Luca's method for bounce detection
                if bounce_count(speed_queue, bbox):
                    bounce += 1
                    pause_frame = True

                    results = mp_results_deque[3]
                    box_center = box_center_deque[3]

                    # foot group
                    r_foot_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                    l_foot_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                    r_ankle_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    l_ankle_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

                    # hip group
                    r_knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                    r_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    l_knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                    l_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

                    # shoulder
                    r_shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    l_shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

                    # head
                    head_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                    r_foot_dist = distance_from_segment(
                        landmark_vector(r_foot_landmark),
                        landmark_vector(r_ankle_landmark),
                        box_center,
                    )
                    l_foot_dist = distance_from_segment(
                        landmark_vector(l_foot_landmark),
                        landmark_vector(l_ankle_landmark),
                        box_center,
                    )
                    r_hip_dist = distance_from_segment(
                        landmark_vector(r_knee_landmark),
                        landmark_vector(r_hip_landmark),
                        box_center
                    )
                    l_hip_dist = distance_from_segment(
                        landmark_vector(l_knee_landmark),
                        landmark_vector(l_hip_landmark),
                        box_center
                    )
                    r_shoulder_dist = distance_from_landmark(
                        landmark_vector(r_shoulder_landmark),
                        box_center
                    )
                    l_shoulder_dist = distance_from_landmark(
                        landmark_vector(l_shoulder_landmark),
                        box_center
                    )
                    head_dist = distance_from_landmark(
                        landmark_vector(head_landmark),
                        box_center
                    )

                    print(f'right foot:\t {r_foot_dist}\n'
                          f'left foot:\t {l_foot_dist}\n'
                          f'right hip:\t {r_hip_dist}\n'
                          f'left hip:\t {l_hip_dist}\n'
                          f'right shou:\t {r_shoulder_dist}\n'
                          f'left shou:\t {l_shoulder_dist}\n'
                          f'head:\t\t {head_dist}')

                    predictions = [r_foot_dist, l_foot_dist, r_hip_dist, l_hip_dist, r_shoulder_dist, l_shoulder_dist, head_dist]
                    predictions_vals = []

                    for val, flg in predictions:
                        if not flg:
                            predictions_vals.append(val*100000)
                        else:
                            predictions_vals.append(val)

                    min_idx = min(range(len(predictions_vals)), key=predictions_vals.__getitem__)
                    predictions_list.append(min_idx)
                    print(f'Predicted: {idx_dict[min_idx]}')
                    cv2.putText(frame,
                                f"{idx_dict[min_idx]}",
                                (20, height - 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (255, 0, 0),
                                2)

            else:
                cv2.putText(frame,
                            "Tracking failure detected",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2)

            # Print tracker and FPS
            cv2.putText(frame,
                        f"{tracker_type} - FPS: {fps:.2f}",
                        (80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 0, 0),
                        2)

            # Print bounce counter
            cv2.putText(frame,
                        f"Bounces: {bounce}",
                        (20, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 0, 0),
                        2)

            # Show image
            cv2.imshow("video", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            if pause_frame:
                time.sleep(1.0)
                pause_frame = False

        log(f'Processed {num_iter} frames. Bounces: {bounce}')
        log(f'Ground th indexes: {real_values_list}')
        log(f'Predicted indexes: {predictions_list}')
