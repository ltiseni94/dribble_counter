import os
import subprocess
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', '-t', type=str, default='CSRT')
    return parser.parse_args()


def main():
    tracker = parse_args().tracker
    videos = ('resources/' + video for video in os.listdir('resources') if video.endswith('.mp4'))
    for video in videos:
        subprocess.run(f'python app.py -t {tracker} -s {video} --save', shell=True, cwd='./')


if __name__ == '__main__':
    main()
