import pandas as pd
import os
import cv2
import numpy as np
import random
from chip.video_libs import global_tranformation, reformatting, high_downscale_compress
import shutil

def apply_transform():

    DATA_PATH = r"E:\STVD_DL\root_data"

    # input_path, input_file, output_path, output_file
    INPUT_ROOT = r"E:\STVD_DL\seta_segment"
    OUTPUT_ROOT = r"E:\STVD_DL\added_video"
    total= 0
    for i in range(0, 243):
        classIDx = str(i)
        root_path = os.path.join(DATA_PATH, classIDx)
        OUTPUT_PATH = os.path.join(OUTPUT_ROOT, classIDx)   
        INPUT_PATH = os.path.join(INPUT_ROOT, classIDx)
        lst = os.listdir(root_path) 
        number_files = len(lst)
        number_remain = 10 - number_files        
        if (number_files < 10):
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            # print(classIDx)
            lst_selected = random.choices(lst, k=number_remain)
            # print(classIDx, len(lst_selected))
            count = 1
            for item in lst_selected:
                output_file = item.split('.')[0] + "_" + str(count) + ".mp4"
                # print(INPUT_PATH, item, OUTPUT_PATH, output_file)
                # global_tranformation(INPUT_PATH, item, OUTPUT_PATH, output_file)
                high_downscale_compress(INPUT_PATH, item, OUTPUT_PATH, output_file)
                count += 1
            total += number_remain
    
    print("Added {} files".format(total))

def post_processing():

    INPUT_ROOT = r"E:\STVD_DL\added_video"
    OUTPUT_ROOT = r"E:\STVD_DL\root_data"
    total = 0

    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                parent_dir = str(root).split('\\')[-1]
                # print(parent_dir, file)
                INPUT_PATH = os.path.join(INPUT_ROOT, parent_dir)
                OUTPUT_PATH = os.path.join(OUTPUT_ROOT, parent_dir)
                # print(INPUT_PATH, file, OUTPUT_PATH, file)
                # reformatting(INPUT_PATH, file, OUTPUT_PATH, file)


                shutil.copy(os.path.join(INPUT_PATH, file), os.path.join(OUTPUT_PATH, file))
                total +=1
        

    print("Processed {} files".format(total))

def get_positive_metadata():

    INPUT_ROOT = r"E:\STVD_DL\root_data"
    data = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                parent_dir = str(root).split('\\')[-1]
                # print(parent_dir, file)
                VIDEO_PATH = os.path.join(root, file)

                cap = cv2.VideoCapture(VIDEO_PATH)
                fps = np.round(cap.get(cv2.CAP_PROP_FPS), 2)
                width, height = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = round(frame_count/fps, 2)
                cap.release()
                case = {
                    'video_name': str(file),
                    'classIDx': parent_dir,
                    'duration_sec': duration,
                    'frame_per_sec': fps,
                    'width': width,
                    'height': height
                }
                data.append(case)

    # video_name, classIDx, duration_sec, frame_per_sec, width, height 
    positive_file = "positive_metadata.csv"
    DATA_PATH = r'data'
    ouput_file = os.path.join(DATA_PATH, positive_file)
    df = pd.DataFrame(data)
    df.to_csv(ouput_file, index=False, header=True)

def get_negative_metadata():

    INPUT_ROOT = r"E:\STVD\stvd\setc\negative"
    data = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                parent_dir = "243"                
                VIDEO_PATH = os.path.join(root, file)

                cap = cv2.VideoCapture(VIDEO_PATH)
                fps = np.round(cap.get(cv2.CAP_PROP_FPS), 2)
                width, height = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = round(frame_count/fps, 2)
                cap.release()
                case = {
                    'video_name': str(file),
                    'classIDx': parent_dir,
                    'duration_sec': duration,
                    'frame_per_sec': fps,
                    'width': width,
                    'height': height
                }
                data.append(case)

    # video_name, classIDx, duration_sec, frame_per_sec, width, height 
    positive_file = "negative_metadata.csv"
    DATA_PATH = r'data'
    ouput_file = os.path.join(DATA_PATH, positive_file)
    df = pd.DataFrame(data)
    df.to_csv(ouput_file, index=False, header=True)


def main():

    # apply_transform()
    # post_processing()
    # get_positive_metadata()
    # get_negative_metadata()

    print()

if __name__ == '__main__':
    main()
    