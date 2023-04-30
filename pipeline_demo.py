from src import *
import argparse
import subprocess
from scene_demo import run_scene_pipe
import shutil
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', help='Path to Scene to be expanded.',default='./data/01_raw/atla_s1e1.mp4')
parser.add_argument('--out_path',help='Folder for output to be stored.',default='./data/03_final/')

def run_pipe(in_path, out_path):
    # split into scenes at the given path
    detect_scenes(in_path)

    # for each scene run the scene pipe
    scene_ret = []
    scene_dir = './data/02_scenes/'#'./data/temp_scenes/'
    for scene in os.listdir(scene_dir):
        new_scene_path = run_scene_pipe(in_path=(scene_dir+scene), out_path=None)
        scene_ret.append(new_scene_path)

    with open('./scene_list.txt', 'w') as f:
        for file_name in scene_ret:
            f.write(f"{file_name}\n")
        f.close()

    # merge all scenes into one video using ffmpeg
    os.makedirs(out_path,exist_ok=True)
    file = in_path.split('/')[-1][:-4]
    title=file+'_res'

    command = f'ffmpeg -f concat -i ./scene_list.txt -c copy {out_path}/{title}.mp4'
    subprocess.call(command.split(' '),cwd=os.getcwd(),shell=True)

    # delete all scenes and temp scene folder
    shutil.rmtree(scene_dir)
    os.remove('scene_list.txt')

if __name__=='__main__':
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    run_pipe(in_path,out_path)