import cv2
import numpy as np

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