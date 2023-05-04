import os
from multiprocessing import Pool
import pandas as pd

NUM_THREADS = 50
# VIDEO_ROOT = r'E:\STVD\stvd\setd\negative'   # Directory for videos
VIDEO_ROOT = r'E:\STVD_DL\root_data'
FRAME_ROOT = r'E:\STVD_DL\data\val'  # Directory for extracted frames

groundtruth_file = r"data\pos_val_metadata.csv"

def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, classIDx, tmpl='%04d.jpg'):
    '''
    video[-4]: to get the name of video without .mp4
    FPS = 1 for positive
    FPS = 0.08 for negative
    '''
    tmpl = video[:-4] + '_%d.jpg'
    str_command = f'ffmpeg -nostats -loglevel 0 -i {VIDEO_ROOT}/{classIDx}/{video} -vf fps=12 ' f'{FRAME_ROOT}/{classIDx}/{tmpl}'
    # str_command = f'ffmpeg -nostats -loglevel 0 -i {VIDEO_ROOT}/{video} -vf fps=0.08 ' f'{FRAME_ROOT}/{classIDx}/{tmpl}'
    # print(str_command
    os.system(str_command)


def target(video_list):

    for i in range(len(video_list)):
        video_name = str(video_list[i][0])
        classIDx = str(video_list[i][1])
        # print(video_name, classIDx)
        # video_path = os.path.join(FRAME_ROOT, classIDx)
        # if not os.path.exists(video_path):
        #     os.makedirs(video_path)        
        extract(video_name, classIDx)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)


df = pd.read_csv(groundtruth_file)

splits = list(split(df.values.tolist(), NUM_THREADS))


if __name__ == '__main__':
    print("[INFO] starting .....")
    video_lst = df.values.tolist()
    for i in range(len(video_lst)):
        video_name = str(video_lst[i][0])
        classIDx = str(video_lst[i][1])
        # print(video_name, classIDx)
        video_path = os.path.join(FRAME_ROOT, classIDx)
        if not os.path.exists(video_path):
            os.makedirs(video_path)

    p = Pool(NUM_THREADS)
    p.map(target, splits)
    p.close()
    p.join()