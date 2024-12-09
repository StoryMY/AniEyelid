import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--both', action='store_true')
    args = parser.parse_args()

    mask_dir = os.path.join(args.dir, 'mask')
    eyemask_dir = os.path.join(args.dir, 'eyemask')
    newmask_dir = os.path.join(args.dir, 'newmask')
    if args.both:
        eyemask2_dir = os.path.join(args.dir, 'eyemask2')

    os.makedirs(newmask_dir, exist_ok=True)

    if not os.path.exists(mask_dir) or not os.path.exists(eyemask_dir):
        print('No directory! Quit!')
        exit(0)

    mask_list = os.listdir(mask_dir)
    mask_list.sort()
    for mask_name in tqdm(mask_list):
        mask_path = os.path.join(mask_dir, mask_name)
        eyemask_path = os.path.join(eyemask_dir, mask_name)
        newmask_path = os.path.join(newmask_dir, mask_name)

        mask = cv2.imread(mask_path, -1) / 255
        eyemask = cv2.imread(eyemask_path)[:, :, 0] / 255
        
        if args.both:
            eyemask2_path = os.path.join(eyemask2_dir, mask_name)
            eyemask2 = cv2.imread(eyemask2_path)[:, :, 0] / 255
            newmask = (1 - eyemask) * (1 - eyemask2) * mask
            newmask = (newmask * 255).astype(np.uint8)
        else:
            newmask = (1 - eyemask) * mask
            newmask = (newmask * 255).astype(np.uint8)
        cv2.imwrite(newmask_path, newmask)
