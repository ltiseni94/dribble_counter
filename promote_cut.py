import os
import shutil


if __name__ == '__main__':
    cut_videos = ['resources/' + video for video in os.listdir('resources') if video.endswith('_cut.mp4')]
    cut_labels = ['resources/' + label for label in os.listdir('resources') if label.endswith('_label_cut.csv')]
    try:
        os.mkdir('__temp__')
    except FileExistsError:
        pass
    for video in cut_videos:
        shutil.copy(video, f'__temp__/{video.split("/")[-1][:-8]}.mp4')
    for label in cut_labels:
        shutil.copy(label, f'__temp__/{label.split("/")[-1][:-14]}_label.csv')
