import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from optparse import OptionParser

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# protoFile = './pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
# weightsFile = './pose/mpi/pose_iter_160000.caffemodel'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}]: {msg}')


def is_bounce(prev, actual):
    return 1 if actual < 0 and prev*actual < 0 else 0


def luca_count(q: deque, box: tuple):
    prev_avg = sum(np.diff(q))
    q.append(box[1] + box[3]/2)
    new_avg = sum(np.diff(q))
    log(f'{new_avg} - {prev_avg} - {is_bounce(prev_avg, new_avg)}')
    return is_bounce(prev_avg, new_avg)


def update_dnn(network: cv2.dnn_Net, img: np.ndarray, wid: int, heig: int):

    network_input_blob = cv2.dnn.blobFromImage(image=img,
                                               scalefactor=1.0/255,
                                               size=(wid, heig),
                                               mean=(0, 0, 0),
                                               swapRB=False,
                                               crop=False)
    network.setInput(network_input_blob)
    network_output = network.forward()
    n_frames, n_points, h, w = network_output.shape[:]

    key_points = []

    for i in range(15):
        prob_map = network_output[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        kp = (int(wid * point[0] / w), int(heig * point[1] / h))

        if prob > 0.7:
            cv2.circle(img=img,
                       center=kp,
                       radius=2,
                       color=(0, 255, 255),
                       thickness=-1,
                       lineType=cv2.FILLED)

            cv2.putText(img=img,
                        text=f'{i}',
                        org=kp,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.2,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)
            key_points.append(kp)

    return img, key_points


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-t', '--tracker', action='store', dest='tracker', default='MIL',
                      help='Choose tracker type among: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW",'
                           ' "GOTURN", "MOSSE", "CSRT"')
    parser.add_option('-b', '--box', action='store', dest='bounding_box', default=[0, 0, 100, 100],
                      help='Specify bounding box as a string: "0, 0, 100, 100"')

    opts, args = parser.parse_args()

    tracker_types = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT')
    tracker_type = opts.tracker

    if tracker_type not in tracker_types:
        log(f'Requested tracker {opts.tracker} is not available. Will default to "MIL"')
        tracker_type = "MIL"

    # mil
    # boosting insommina
    # medianflow veloce, attenzione però
    # csrt buono, attenzione ostacoli
    
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

    # net: cv2.dnn_Net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    #video = cv2.VideoCapture('./resources/kaka_cut.mp4')
    video = cv2.VideoCapture('./resources/marcello.mp4')

    if not video.isOpened():
        log('Could not open video, exiting')
        exit(-1)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log(f'Cap size: ({width}, {height})')

    frame_read, frame = video.read()
    if not frame_read:
        log('Could not read frame, exiting')
        exit(-2)

    cv2.namedWindow('first_frame', 1)

    # frame, key_points = update_dnn(net, frame, width, height)

    orig_frame = frame.copy()
    rectangle_corners = []
    cnt = 0

    def mouse_callback(event, x, y, flags, param):
        global frame, rectangle_corners, cnt
        if event == cv2.EVENT_LBUTTONDOWN and cnt < 2:
            rectangle_corners.append((x, y))
            cnt += 1
            log(f'cnt = {cnt:01d}\t'
                f'x = {x:04d}\t'
                f'y = {y:04d}')
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

    tommaso_bounce = 0
    luca_bounce = 0
    luca_queue = deque([bbox[1]+bbox[3]/2]*5, 5)

    bbox_centers_y = []
    delta = deque(bbox_centers_y, 2)

    bbox_centers_y.append((bbox[1] + round(bbox[3] / 2)))

    num_iter = 0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            frame_read, frame = video.read()
            if not frame_read:
                log('No more frames available')
                break
            num_iter += 1

            # updating method + fps calculation
            timer = cv2.getTickCount()
            tracker_ok, bbox = tracker.update(frame)
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if tracker_ok:
                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                # Luca's method for bounce detection
                # bounce = luca_count(luca_queue, bbox)
                # if bounce:
                #     frame, key_points = update_dnn(net, frame, width, height)
                luca_bounce += luca_count(luca_queue, bbox)

                # Tommaso's method for bounce detection
                bbox_centers_y.append((bbox[1] + round(bbox[3] / 2)))
                if not bbox_centers_y[-1] == bbox_centers_y[-2]:  # evito "stazionamento" palla
                    delta.append(bbox_centers_y[-1] - bbox_centers_y[-2])
                if len(delta) > 1:
                    if delta[1] < 0 and (delta[1] * delta[0]) < 0:
                        # delta < 0 : risalita, (delta[1]*delta[0]) < 0 : cambio di segno
                        tommaso_bounce += 1

            else:
                cv2.putText(frame,
                            "Tracking failure detected",
                            (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2)

            # Write tracker type
            cv2.putText(frame,
                        f'{tracker_type} Tracker',
                        (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (50, 170, 50),
                        2)

            # Print FPS
            cv2.putText(frame,
                        f"FPS: {fps:.2f}",
                        (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (50, 170, 50),
                        2)

            # Print Tommaso's bounce
            cv2.putText(frame,
                        f"Bounce: {tommaso_bounce}",
                        (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (50, 170, 50),
                        2)

            # Print Luca's bounce
            cv2.putText(frame,
                        f"Luca's bounce: {luca_bounce}",
                        (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (50, 170, 50),
                        2)

            # Show image
            cv2.imshow("Kaka", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        log(f'Processed {num_iter} frames. Bounces: {luca_bounce}')
