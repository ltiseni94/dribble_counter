import cv2
import time
from collections import deque
from optparse import OptionParser


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}]: {msg}')


def is_bounce(prev, actual):
    return 1 if actual < 0 and prev*actual < 0 else 0


def diff_queue(q: deque):
    diff = []
    for i in range(len(q)-1):
        diff.append(q[i+1] - q[i])
    return diff


def luca_count(q: deque, box: tuple):
    prev_avg = sum(diff_queue(q))
    q.append(box[1] + box[3]/2)
    new_avg = sum(diff_queue(q))
    return is_bounce(prev_avg, new_avg)


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

    video = cv2.VideoCapture('./resources/kaka_cut.mp4')

    if not video.isOpened():
        log('Could not open video, exiting')
        exit(-1)

    frame_read, frame = video.read()
    if not frame_read:
        log('Could not read frame, exiting')
        exit(-2)

    cv2.namedWindow('first_frame', 1)

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
            cv2.destroyAllWindows()
            break

    x_vals = (rectangle_corners[0][0], rectangle_corners[1][0])
    y_vals = (rectangle_corners[0][1], rectangle_corners[1][1])

    bbox = (min(x_vals), min(y_vals), abs(x_vals[1]-x_vals[0]), abs(y_vals[1]-y_vals[0]))
    log(f'Bounding Box: {bbox}')

    tracker_ok = tracker.init(orig_frame, bbox)

    bounce = 0
    luca_bounce = 0
    luca_queue = deque([bbox[1]+bbox[3]/2]*5, 5)
    bbox_centers = []
    bbox_centers_y = []
    delta = deque(bbox_centers_y, 2)

    bbox_centers_y.append((bbox[1] + round(bbox[3] / 2)))

    num_iter = 0
    while True:
        frame_read, frame = video.read()
        if not frame_read:
            log('No more frames available')
            break
        num_iter += 1

        timer = cv2.getTickCount()
        tracker_ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if tracker_ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            luca_bounce += luca_count(luca_queue, bbox)

            bbox_centers_y.append((bbox[1] + round(bbox[3] / 2)))

            if not bbox_centers_y[-1] == bbox_centers_y[-2]:  # evito "stazionamento" palla
                delta.append(bbox_centers_y[-1] - bbox_centers_y[-2])

            if len(delta) > 1:
                if delta[1] < 0 and (delta[1] * delta[0]) < 0:
                    # delta < 0 : risalita, (delta[1]*delta[0]) < 0 : cambio di segno
                    bounce += 1
        else:
            # Tracking failure
            cv2.putText(frame,
                        "Tracking failure detected",
                        (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2)

        cv2.putText(frame,
                    f'{tracker_type} Tracker',
                    (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 170, 50),
                    2)

        cv2.putText(frame,
                    f"FPS: {fps:.2f}",
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 170, 50),
                    2)

        cv2.putText(frame,
                    f"Bounce: {bounce}",
                    (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 170, 50),
                    2)

        cv2.putText(frame,
                    f"Luca's bounce: {luca_bounce}",
                    (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 170, 50),
                    2)

        cv2.imshow("Kakà", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    log(f'Processed {num_iter} frames')