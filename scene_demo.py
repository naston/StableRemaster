from src import *
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--in_path', help='Path to Scene to be expanded.',default='./data/02_scenes/atla_s1e1-Scene-151.mp4')
parser.add_argument('--out_path',help='Folder for output to be stored.',default='./data/03_final/')

def run_scene_pipe(in_path, out_path):
    # seperate video into frames
    print(in_path)
    frames = split_video(in_path)
    
    # resize?
    if frames[0].size is not (940,720):
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (940,720), interpolation = cv2.INTER_AREA)
            resized_frames.append(resized)
        frames = resized_frames

    # run object detection
    segmenter = background_segmentation_loader()
    masks = []
    bgs = []
    for frame in tqdm(frames):
        bg, bg_mask = segmenter(frame,mask_conf=0.2,cat_conf=0.4)
        masks.append(bg_mask)
        bgs.append(bg)

    # generate total background
    total_bg, total_mask, Ms = stitch_multiple(frames, masks)

    # resample frames
    pipe = get_sd_pipe()
    new_frames = []
    for i in range(len(frames)):
        new_frame, total_bg, total_mask = resample_frame(frames[i],masks[i], Ms[i], total_bg, total_mask, pipe)
        new_frames.append(new_frame)

    # create video output
    os.makedirs(out_path,exist_ok=True)
    file = in_path.split('/')[-1][:-4]
    title=file+'_res'
    create_video(new_frames, f'{out_path}/{title}.avi')

    return f'{out_path}/{title}.avi'

if __name__=='__main__':
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    run_scene_pipe(in_path,out_path)