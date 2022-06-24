import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Video to resize')
    parser.add_argument('--resize', '-r', type=int, default=2, help='Resize factor')
    parser.add_argument('--output', '-o', default=None, help='output path for new video')
    return parser.parse_args()


def main():
    args = parse_args()
    source: str = args.source
    if not source.endswith('.mp4'):
        raise ValueError(f'Incompatible input format "{source[-4:]}". Expected ".mp4"')
    output: str = args.output
    if output is None:
        output = source[:-4] + '_resized.mp4'

    r_factor = args.resize
    if type(r_factor) is not int:
        raise TypeError('input argument resize is not an integer')
    if r_factor <= 1:
        raise ValueError('Cannot enlarge video')

    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise IOError('Could not create and open VideoCapture')

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // r_factor
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) // r_factor
    frame_fps = int(video.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (width, height))

    while True:
        res, frame = video.read()
        if not res:
            break
        frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)


if __name__ == '__main__':
    main()
