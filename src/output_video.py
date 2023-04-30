import cv2
import numpy as np

def create_video(frames, out_path):
    height, width, layers = frames[0].shape
    size = (width,height)
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for frame in frames:
        out.write(frame)

    out.release()