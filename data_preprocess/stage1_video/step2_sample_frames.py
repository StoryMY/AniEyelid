import cv2
import os
import numpy as np
import argparse
from utils import smart_mkdir

def sharpness(image_path):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

def select_sharpest_image(img_path_list):
    sv_arr = []
    for i, img_path in enumerate(img_path_list):
        sv = sharpness(img_path)
        sv_arr.append((i, sv))
    
    sv_arr = sorted(sv_arr, key=lambda x: x[1], reverse=True)

    return sv_arr[0]    # return index


def main(args):
    image_dir, sparse_dir, work_dir = smart_mkdir(args.path)

    imglist = os.listdir(image_dir)
    imglist.sort()

    imglist_sta = imglist[:args.split]
    imglist_dyn = imglist[args.split:]

    selected_sv = []

    ctr = 0

    # Part 1 (participant keep still, while camera moving in the front)
    sta_idx = 0
    while True:
        st = sta_idx * args.gap1
        ed = st + args.gap1
        if ed > len(imglist_sta):
            break
        img_path_list = [os.path.join(image_dir, item) for item in imglist_sta[st:ed]]
        idx, sv = select_sharpest_image(img_path_list)
        selected_sv.append(sv)
        in_path = os.path.join(image_dir, imglist_sta[st + idx])
        out_path = os.path.join(sparse_dir, '%03d.png' % ctr)
        os.system('copy %s %s' % (in_path, out_path))   # Windows
        sta_idx += 1
        ctr += 1

    # Part 2 (camera keep still, while participant performing different gazes)
    dyn_idx = 0
    while True:
        st = dyn_idx * args.gap2
        ed = st + args.gap2
        if ed > len(imglist_dyn):
            break
        img_path_list = [os.path.join(image_dir, item) for item in imglist_dyn[st:ed]]
        idx, sv = select_sharpest_image(img_path_list)
        selected_sv.append(sv)
        in_path = os.path.join(image_dir, imglist_dyn[st + idx])
        out_path = os.path.join(sparse_dir, '%03d.png' % ctr)
        os.system('copy %s %s' % (in_path, out_path))   # Windows
        dyn_idx += 1
        ctr += 1

    min_sv, max_sv = np.min(selected_sv), np.max(selected_sv)
    mean_sv, std_sv = np.mean(selected_sv), np.std(selected_sv, ddof=1)
    with open(os.path.join(work_dir, 'sharpness.txt'), 'w') as f:
        f.write('%f %f\n' % (min_sv, max_sv))
        f.write('%f %f\n' % (mean_sv, std_sv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='The video path')
    parser.add_argument('--gap1', type=int, default=5, help='Frame gap for Part 1')
    parser.add_argument('--gap2', type=int, default=5, help='Frame gap for Part 2')
    parser.add_argument('--split', type=int, default=1500, help='Split index for Part 1 and Part 2')
    args = parser.parse_args()
    if args.gap2 == -1:
        args.gap2 = args.gap1
    print(args)

    main(args)
