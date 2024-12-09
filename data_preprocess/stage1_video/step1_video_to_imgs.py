import cv2
import os
import numpy as np
import argparse
from utils import smart_mkdir

def convert_video_to_imgs(video_path):
    image_dir, _, _ = smart_mkdir(video_path)
    cmd = "ffmpeg -i %s -start_number 0 %s" % (video_path, image_dir + f'/%04d.png')
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='The video path')
    args = parser.parse_args()
    convert_video_to_imgs(args.path)
