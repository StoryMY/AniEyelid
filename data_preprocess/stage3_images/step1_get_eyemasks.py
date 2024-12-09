import os
import cv2
import numpy as np
import argparse
from utils import generate_seg, read_lmk
from tqdm import tqdm


def single_image(img_path, lmk_path, mask_path, imask_path=None, mask2_path=None, imask2_path=None, manual_mask=False):
    img = cv2.imread(img_path)
    face_lmk, eyelid_lmk, iris_lmk = read_lmk(lmk_path)

    # image-right eye
    if not manual_mask:
        seg_lid_r = generate_seg(img, eyelid_lmk[24:], True)
        cv2.imwrite(mask_path, seg_lid_r)
    if imask_path is not None:
        seg_iris_r = generate_seg(img, iris_lmk[19:], False)
        cv2.imwrite(imask_path, seg_iris_r)

    # image-left eye
    if mask2_path is not None:
        seg_lid_l = generate_seg(img, eyelid_lmk[:24], True)
        cv2.imwrite(mask2_path, seg_lid_l)
    if imask2_path is not None:
        seg_iris_l = generate_seg(img, iris_lmk[:19], False)
        cv2.imwrite(imask2_path, seg_iris_l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--both', action='store_true')
    args = parser.parse_args()

    img_dir = os.path.join(args.dir, 'rgb')
    lmk_dir = os.path.join(args.dir, 'landmark')
    eyemask_dir = os.path.join(args.dir, 'eyemask')
    irismask_dir = os.path.join(args.dir, 'irismask')
    os.makedirs(eyemask_dir, exist_ok=True)
    os.makedirs(irismask_dir, exist_ok=True)
    if args.both:
        eyemask2_dir = os.path.join(args.dir, 'eyemask2')
        irismask2_dir = os.path.join(args.dir, 'irismask2')
        os.makedirs(eyemask2_dir, exist_ok=True)
        os.makedirs(irismask2_dir, exist_ok=True)


    img_list = os.listdir(img_dir)
    img_list.sort()
    lmk_list = os.listdir(lmk_dir)
    lmk_list.sort()
    for idx, img_name in enumerate(tqdm(img_list)):
        img_path = os.path.join(img_dir, img_name)
        lmk_path = os.path.join(lmk_dir, lmk_list[idx])
        eyemask_path = os.path.join(eyemask_dir, img_name[:-4] + '.png')
        irismask_path = os.path.join(irismask_dir, img_name[:-4] + '.png')
        if args.both:
            eyemask2_path = os.path.join(eyemask2_dir, img_name[:-4] + '.png')
            irismask2_path = os.path.join(irismask2_dir, img_name[:-4] + '.png')
        else:
            eyemask2_path = None
            irismask2_path = None

        single_image(img_path, lmk_path, eyemask_path, irismask_path, eyemask2_path, irismask2_path, manual_mask=False)
