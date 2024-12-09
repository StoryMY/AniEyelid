import os
import sys
import argparse
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from colmap.read_write_model import read_cameras_binary, read_points3d_binary, read_images_binary, write_images_binary, write_cameras_binary, write_points3d_binary
import colmap.read_write_model as rwm

def load_camera_data(camera_path):
    camera_data = read_cameras_binary(camera_path)
    list_of_keys = list(camera_data.keys())
    cam = camera_data[list_of_keys[0]]
    h, w = cam.height, cam.width
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    print('imgH', 'imgW', 'f', 'cx', 'cy')
    print('Load:', h, w, f, cx, cy)
    return h, w, f, cx, cy

def save_camera_data(save_path, h, w, f, cx, cy):
    new_cam_item = rwm.Camera(id=1, model='SIMPLE_PINHOLE', 
                        width=int(w), height=int(h),
                        params=np.array([f, cx, cy]))
    new_cam = {}
    new_cam[1] = new_cam_item
    write_cameras_binary(new_cam, save_path)


def process_img(img, out_w, out_h, dx, dy, resize, is_mask=False):
    new_img = img[dy:dy+out_h, dx:dx+out_w]
    if resize != -1:
        new_img = cv2.resize(new_img, (resize, resize))
        if is_mask:
            new_img[new_img > 128] = 255
            new_img[new_img <= 128] = 0
    return new_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--crop', type=int, nargs='+', default=[1080, 1080, 0, 400], help='out_w:out_h:x:y')
    parser.add_argument('--resize', type=int, default=-1, help='assume the image is square, -1: no resize')
    parser.add_argument('--name', type=str, default='crop')
    parser.add_argument('--go', default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    # Camera Intrinsics
    camera_path = os.path.join(args.path, 'sparse', '0', 'cameras.bin')
    h, w, f, cx, cy = load_camera_data(camera_path)
    out_w, out_h, dx, dy = args.crop
    cx = cx - dx
    cy = cy - dy
    w = out_w
    h = out_h
    if args.resize != -1:
        assert w == h, (w, h)
        scale = args.resize / w
        w = args.resize
        h = args.resize
        cx *= scale
        cy *= scale
        f *= scale
    print('Save:', h, w, f, cx, cy)
    save_dir = args.path.strip('/') + '_' + args.name
    print(save_dir)
    if not args.go:
        exit(0)
    os.makedirs(os.path.join(save_dir, 'sparse', '0'), exist_ok=True)
    save_camera_data(os.path.join(save_dir, 'sparse', '0', 'cameras.bin'), h, w, f, cx, cy)
    shutil.copyfile(os.path.join(args.path, 'sparse', '0', 'images.bin'), os.path.join(save_dir, 'sparse', '0', 'images.bin'))
    shutil.copyfile(os.path.join(args.path, 'sparse', '0', 'points3D.bin'), os.path.join(save_dir, 'sparse', '0', 'points3D.bin'))

    # Image
    select_dir_list = ['images', 'rgb', 'mask', 'newmask', 'eyemask', 'irismask', 'eyemask2', 'irismask2']
    for item in select_dir_list:
        img_dir = os.path.join(args.path, item)
        if not os.path.exists(img_dir):
            continue
        print(img_dir)
        is_mask = item.find('mask') != -1
        print('is_mask', is_mask)
        os.makedirs(os.path.join(save_dir, item), exist_ok=True)
        img_list = os.listdir(img_dir)
        img_list.sort()
        for img_name in tqdm(img_list):
            img_path = os.path.join(img_dir, img_name)
            save_path = os.path.join(save_dir, item, img_name)
            img = cv2.imread(img_path, -1)
            new_img = process_img(img, out_w, out_h, dx, dy, args.resize, is_mask)
            cv2.imwrite(save_path, new_img)

    # Other Files
    file_list = ['sparse_points_interest.ply']
    for file in file_list:
        if os.path.exists(os.path.join(args.path, file)):
            shutil.copyfile(os.path.join(args.path, file), os.path.join(save_dir, file))
