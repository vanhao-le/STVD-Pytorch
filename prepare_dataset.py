import pandas as pd
import os
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np


DATA_PATH = Path("data/")
GROUNDTRUTH_FILE = DATA_PATH / "groundtruth.csv"
REFERENCE_FILE =  DATA_PATH / "query_stvd.csv"
CATEGORY_FILE = DATA_PATH / "category.csv"
DUPLICATE_STATISTIC_FILE = DATA_PATH / "near_duplicate_stats.csv"

def generate_category():
    VIDEO_PATH = r'E:\STVD\stvd\references'
    df = pd.read_csv(REFERENCE_FILE)
    data = []
    for idx, item in df.iterrows():
        class_name = item['Reference_Video']
        video_path = os.path.join(VIDEO_PATH, class_name)
        cap = cv2.VideoCapture(video_path)
        fps = np.round(cap.get(cv2.CAP_PROP_FPS), 2)        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count/fps, 3)
        class_name = str(class_name).split('.')[0]
        case = {
            'class_name': class_name,
            'classIDx': idx,
            'duration_sec': duration,
        }
        data.append(case)
    
    df = pd.DataFrame(data)
    df.to_csv(CATEGORY_FILE, index=False, header=True)

def near_duplicate_statistic():
    df = pd.read_csv(GROUNDTRUTH_FILE)
    data = {}
    for idx, item in df.iterrows():
        pos_video = str(item['Positive_Video'])
        ref_video = str(item['Reference_Video'])
        if data.get(ref_video) is None:
            data[ref_video] = 1
        else:   
            data[ref_video] += 1
    
    data = OrderedDict(sorted(data.items(), key=lambda t: t[1]))

    d_lst = []
    for key, value in data.items():
        case = {'reference_video': key, 'no_duplicate': value}
        d_lst.append(case)

    df = pd.DataFrame(d_lst)
    df.to_csv(DUPLICATE_STATISTIC_FILE, index=False, header=True)

    

def split_single_video(input_video, output_video, start_num, end_num):

    """
    cmd: ffmpeg -nostats -loglevel 0 -i input.mp4 -vf select="between(n\,2405\,2516),setpts=PTS-STARTPTS"
    -c:v libx264 -x264-params \"nal-hrd=cbr\" -b:v "29k" -minrate "29k" -maxrate "29k" -bufsize "32k" out.mp4

    index from 0
    """

    str_cmd = "ffmpeg -nostats -loglevel 0 -i " +  input_video \
        + " -vf select=\"between(n\," + start_num + "\," + end_num \
        + "),setpts=PTS-STARTPTS\" -c:v libx264 -x264-params \"nal-hrd=cbr\" -b:v \"15k\" -minrate \"15k\" -maxrate \"15k\" -bufsize \"16k\" "\
        + output_video

    # print(str_cmd)
    os.system(str_cmd)

def generate_video():

    df = pd.read_csv(GROUNDTRUTH_FILE)
    category_df = pd.read_csv(CATEGORY_FILE)

    INPUT_PATH = r"E:\STVD\stvd\setc\positive"
    OUTPUT_PATH = r"E:\STVD_DL\root_data"

    # INPUT_PATH = r"E:\STVD\stvd\seta\positive"
    # OUTPUT_PATH = r"E:\STVD_DL\seta_segment"

    for idx, item in df.iterrows():
        ref_video = str(item['Reference_Video'])
        class_name = ref_video.split('.')[0]
        positive_video = str(item['Positive_Video'])
        ref_len = int(item['Reference_Length'])
        pos_len = int(item['Positive_Length'])
        start_copy = int(item['Start_Copy']) - 1
        end_copy = start_copy + ref_len
        classIDx = category_df[category_df['class_name'] == class_name].values[0][1]
        # print(ref_video, classIDx)
        output_path = os.path.join(OUTPUT_PATH, str(classIDx))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        output_file = os.path.join(output_path, positive_video)
        pre_name = positive_video.split('.')[0]
        for r, d, f in os.walk(INPUT_PATH):
            for file in f:
                if file.startswith(pre_name) and file.endswith(".mp4"):
                    input_file = os.path.join(INPUT_PATH, file)
                    split_single_video(input_file, output_file, str(start_copy), str(end_copy))
        


def main():

    generate_category()

    # near_duplicate_statistic()

    # generate_video()

    

    
    print()
if __name__ == '__main__':
    main()
    



