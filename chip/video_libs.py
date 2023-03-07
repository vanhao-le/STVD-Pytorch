import os
import random
import cv2
import imutils
import numpy as np
from chip.utils import random_walk

"""
set B
alpha: 0.25, 0.5, 0.75, 0.9
1/beta: 40, 20, 10, 5
resolution: 80x60, 160x120, 240x180, 288x216
compress (kbps): 14k, 28k, 56k, 112k
"""
def low_downscale_compress(input_path, input_file, output_path, type_T):

    buff = "112k"
    if(type_T == "B"):
        method = random.randint(1, 4)
        if(method == 1):
            compress = "14k"
            resolution = "80x60"
        if(method == 2):
            compress = "28k"
            resolution = "160x120"
        if(method == 3):
            compress = "56k"
            resolution = "240x180"
        if(method == 4):
            compress = "112k"
            resolution = "288x216"
    
    input_video = input_path + "\\" + input_file
    new_filename = output_path + "\\" +input_file.strip().split('.')[0] + "_" + str(type_T + str(method)) + ".mp4"

    str_command = "ffmpeg -nostats -loglevel 0 -i " + input_video + " -s " + str(resolution) \
                + " -c:v libx264 -x264-params \"nal-hrd=cbr\" -b:v " + compress  \
                + " -minrate " + compress + " -maxrate " + compress + " -bufsize " + buff  \
                + " " + new_filename
    
    os.system(str_command)    
# -------------------------------------------------------------------------------------------------------------

"""
set C
alpha: 0.1, 0.2, 0.25
1/beta: 40, 60, 80
resolution: 32x24, 64x48, 80x60
compress (kbps): 7k, 10k, 14k
"""
def high_downscale_compress(input_path, input_file, output_path, output_file):

    buff = "14k"    
    method = random.randint(1, 4)
    if(method == 1):
        compress = "7k"
        resolution = "32x24"
    if(method == 2):
        compress = "10k"
        resolution = "64x48"
    if(method == 3):
        compress = "14k"
        resolution = "80x60"
    if(method == 4):
        compress = "14k"
        resolution = "32x24"
    
    input_video = os.path.join(input_path, input_file)
    # new_filename = output_path + "\\" +input_file.strip().split('.')[0] + "_" + str(type_T + str(method)) + ".mp4"
    new_filename = os.path.join(output_path, output_file)

    str_command = "ffmpeg -nostats -loglevel 0 -i " + input_video + " -s " + str(resolution) \
                + " -c:v libx264 -x264-params \"nal-hrd=cbr\" -b:v " + compress  \
                + " -minrate " + compress + " -maxrate " + compress + " -bufsize " + buff  \
                + " " + new_filename
    
    os.system(str_command)    
# ----------------------------------------------------------------------------------------------------

"""
fliping video

FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org. 
It is platform dependent. Following codecs work fine: In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. 
( XVID is more preferable. MJPG results in high size video. X264 gives very small size video ) 
In Windows: DIVX ( more to be tested and added )

"""
def flipping(input_path, input_file, output_path, output_file):
       
    video_file = os.path.join(input_path, input_file)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))
    # size = (192, 144)
   
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_name = os.path.join(output_path,output_file)
    
    out = cv2.VideoWriter(output_name, fourcc, fps, size)
        
    while(True):
        ret, frame = cap.read()
        if ret:            
            image = cv2.flip(frame, 1)
            out.write(image)
        else:            
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
# ----------------------------------------------------------------------------------------------------
def rotating(input_path, input_file, output_path, output_file):
    
    input_video = os.path.join(input_path, input_file)

    value = random.randint(0, 3)
    if(value == 0):
         degree = 0         
    if(value == 1):
        degree = 90        
    if(value == 2):
        degree = 180
    if(value == 3):
        degree = 270
    
    out_filename = output_file
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = os.path.join(output_path, out_filename)
    out = cv2.VideoWriter(output, fourcc, fps, size)
    
    while(True):
        ret, frame = cap.read()
        if ret:
            image = imutils.rotate_bound(frame, degree)
            rotated = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            out.write(rotated)
        else:            
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return
# ----------------------------------------------------------------------------------------------------
def add_blackboder(input_path, input_file, output_path, output_file):
    
    input_video = os.path.join(input_path, input_file)
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))
    
    ratio_list = [0.46, 0.56, 0.63, 0.75, 1.33, 1.6, 1.78, 2.17]
    value = np.random.choice(ratio_list)
    
    new_w, new_h = width, height
    # ratio: 0.46, 0.56, 0.63, 0.75, 1.33, 1.6, 1.78, 2.17
    if(value == 0.46):
        new_w, new_h = 110, 240
    if(value == 0.56):
        new_w, new_h = 134, 240
    if(value == 0.63):
        new_w, new_h = 152, 240
    if(value == 0.75):
        new_w, new_h = 180, 240
    if(value == 1.33):
        new_w, new_h = 320, 240
    if(value == 1.6):
        new_w, new_h = 320, 200
    if(value == 1.78):
        new_w, new_h = 320, 180
    if(value == 2.17):
        new_w, new_h = 320, 148
    
    out_filename = output_file
    
    size_horizontal, size_vertical = 0, 0
    if(value < 1):        
        size_vertical = int( (width - new_w)  / 2) 
    else:        
        size_horizontal = int( (height - new_h) / 2)

    original_size = width, height
       
    # print(size, size_horizontal, size_vertical)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = os.path.join(output_path, out_filename)
    out = cv2.VideoWriter(output, fourcc, fps, original_size)
    size = (new_w, new_h)
    
    # slicing arrray
    # arr[0:2][0:2]
    color = [0, 0, 0] # 'black'

    while(True):
        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, size , interpolation = cv2.INTER_AREA)

            if(value < 1):
                top, bottom, left, right = 0, 0, size_vertical, size_vertical
                img_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            else:
                top, bottom, left, right = size_horizontal, size_horizontal, 0, 0
                img_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)            
            # print(image.shape)
            # final_frame = cv2.resize(img_with_border, original_size, cv2.INTER_AREA)
            out.write(img_with_border)
        else:            
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
# ----------------------------------------------------------------------------------------------------

def global_tranformation(input_path, input_file, output_path, output_file):    
    # set D
    # flipping: Yes/no
    # rotating: 
    # black border: 

    #returns a number between 1 and 3 (both included)
    method = random.randint(1, 3)
    if(method == 1):
        flipping(input_path, input_file, output_path, output_file)
    if(method == 2):
        rotating(input_path, input_file, output_path,output_file)
    if(method == 3):
        add_blackboder(input_path, input_file, output_path, output_file)            
    return
# ----------------------------------------------------------------------------------------------------

"""
post-processing video to reduce the video bitrate and the resolution
alpha: 0.6
beta: 20
resolution: 192x144
bitrate: 28k
"""
def reformatting(input_path, input_file, output_path, output_file):
    resolution = "192x144"
    compress = "28k"
    buff = "32k"
    input_video = os.path.join(input_path, input_file)
    new_filename = os.path.join(output_path, output_file)

    str_command = "ffmpeg -nostats -loglevel 0 -i " + input_video + " -s " + str(resolution) \
                + " -c:v libx264 -x264-params \"nal-hrd=cbr\" -b:v " + compress  \
                + " -minrate " + compress + " -maxrate " + compress + " -bufsize " + buff  \
                + " " + new_filename
    
    os.system(str_command)
    return

"""
simulate the video speeding using FPS drops

"""
def video_speeding(input_path, input_file, output_path, output_file, start_copy, copy_length):

    fps_list = random_walk()
    start_copy = start_copy
    copy_length = copy_length

    video_path = os.path.join(input_path, input_file)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get( cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)))
    max_step = int(copy_length / fps)

  
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = os.path.join(output_path, output_file)

    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    step = 0
    frame_list = []
    while success and frame_number < frame_total:
        if step >= max_step:
            frame_list.append(frame)
            out.write(frame)
            frame_number += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
        
        elif frame_number == start_copy:
            fps_new = fps_list[step]
            step += 1
            start_copy += 30
            interval = 30 - fps_new
            for i in range(interval):
                frame_list.append(frame) 
                out.write(frame)

            frame_number += interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
        
        else:
            frame_list.append(frame)
            out.write(frame)
            frame_number += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
