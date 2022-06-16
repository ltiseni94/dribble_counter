# Dribble counter

A Python application capable of counting how many dribbles you do, and with which part
of the body.

## Input sources

- Recorded video in various formats (.mp4, .avi, ...)
- Webcams

## Trackers

- Can use any openCV trackers
- Our custom HSV tracker is available
- Supports your custom tracker if compliant with the openCV tracking API

## How to install it

- Tested on MacOS and Ubuntu 20.04 LTS

Clone the repository:
    
    $ git clone https://github.com/ltiseni94/dribble_counter.git

Go into the folder and create the virtual environment:
    
    $ cd dribble_counter
    $ python3 -m venv venv

Source virtual environment and install dependencies:
    
    $ source venv/bin/activate
    (venv) $ python -m pip install -U pip
    (venv) $ pip install -r requirements.txt

Now you are ready to run the app. Try it with our demo video:

    (venv) $ python app.py

Draw the bounding box around the ball and see if it works!

### Try Custom Hue Tracker

    (venv) $ python app.py -t HUE

Draw the bounding box and then set values for the mask with the interactive
mask view. Ensure the ball is at least 50% visible trying to exclude most of the
background. Good values are:
- Low H = 0
- High H = 95
- Low S = 45
- High S = 255
- Low V = 0
- High V = 255

## Print usage to see available config options
    
    (venv) $ python app.py -h

You should see this output:

    usage: app.py [-h] [-t TRACKER] [-b BBOX] [-q QUEUE] [-r RESIZE] [-s SOURCE] [--bs BS] [--be BE] [--ps PS]
              [--pe PE] [--mindetection MINDETECTION] [--mintracking MINTRACKING] [--pause]

    Dribble detector application. Open a video source (camera or a video file) Draw a bounding box centered on the
    ball with which the user want to do dribbles. Then the app will start counting dribbles. Using Pose tracking by
    MediaPipe, the app will detect the body segment with which the user perform the dribble
    
    optional arguments:
      -h, --help            show this help message and exit
      -t TRACKER, --tracker TRACKER
                            Choose tracker type among: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN",
                            "MOSSE", "CSRT", "HUE"
      -b BBOX, --bbox BBOX  Specify the starting bounding box for the ball through command line
      -q QUEUE, --queue QUEUE, --queue-size QUEUE
                            Queue size for average values calculation - filtering
      -r RESIZE, --resize RESIZE, --resize-input RESIZE
                            Decrease size of video input by specified scale factor
      -s SOURCE, --source SOURCE
                            Video source. Specify a path to a video or a camera
      --bs BS, --ball-start BS
                            specify start index for average ball position value when a dribble is detected
      --be BE, --ball-end BE
                            specify end index for average ball position value when a dribble is detected
      --ps PS, --pose-start PS
                            specify start index for average pose position value when a dribble is detected
      --pe PE, --pose-end PE
                            specify end index for average pose position value when a dribble is detected
      --mindetection MINDETECTION, --min-detection-confidence MINDETECTION
                            min detection confidence for mediapipe pose
      --mintracking MINTRACKING, --min-tracking-confidence MINTRACKING
                            min tracking confidence for mediapipe pose
      --pause, --pause-frame
                            Stop frame for a second when a bounce is detected
      --record-output       Save output video in mp4 format
