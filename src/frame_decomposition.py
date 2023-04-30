import cv2
import numpy as np
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os
import ffmpeg

def split_video(video_file, save_fps=30):
    
    # read in video   
    cap = cv2.VideoCapture(video_file)
    # get video FPS
    fps = min(cap.get(cv2.CAP_PROP_FPS), save_fps)

    # start the loop
    frames = []

    while True:
        is_read, frame = cap.read()
        if is_read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    return frames

def detect_scenes(in_path):
    cwd = os.getcwd()

    os.makedirs('./data/temp_scenes',exist_ok=True)
    os.chdir('./data/temp_scenes')

    scene_list = detect(in_path, ContentDetector(),show_progress=True)

    title = in_path.split('/')[-1][:-4]
    split_video_ffmpeg(in_path, scene_list, video_name=title, show_progress=True)

    os.chdir(cwd)