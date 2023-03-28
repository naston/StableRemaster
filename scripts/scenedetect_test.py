from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os

cwd = os.getcwd()
os.chdir('./data/02_scenes')

indir = '../01_raw'
print('Calculating Scenes...')
scene_list = detect(f'{indir}/s1e1_alt.mp4', ContentDetector(),show_progress=True)

print('Creating Scenes...')
split_video_ffmpeg(f'{indir}/s1e1_alt.mp4', scene_list,video_name='atla_s1e1', show_progress=True)