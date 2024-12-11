import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import cv2

from utils import to_tensor, load_image
from render import CameraInfo
from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3d_binary

def read_model(path, ext=".bin"):
    cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
    images = read_images_binary(os.path.join(path, "images" + ext))
    points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def read_txt(path):
    ret = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ret.append(int(line.strip()))
    return ret


class ColmapDataset(Dataset):
    def __init__(self, data_config, load_img=True):
        root_dir = data_config['root_dir']
        self.select_list = data_config['select']
        if load_img:
            self.images = self.load_dir(os.path.join(root_dir, data_config['img']), self.select_list)
            self.iris_masks = self.load_dir(os.path.join(root_dir, data_config['iris']), self.select_list)
            self.eye_masks = self.load_dir(os.path.join(root_dir, data_config['eye']), self.select_list)
            self.n_images = len(self.images)
        else:
            self.n_images = data_config['dummy_img_num']
            self.n_images_all = data_config['dummy_img_num']
            self.images = [np.zeros([1920, 1080, 3], dtype=np.uint8) for _ in range(self.n_images)]
            self.iris_masks = [np.zeros([1920, 1080, 1], dtype=np.uint8) for _ in range(self.n_images)]
            self.eye_masks = [np.zeros([1920, 1080, 1], dtype=np.uint8) for _ in range(self.n_images)]
        self.get_camera_details(os.path.join(root_dir, data_config['camera']), self.select_list)
        
        if self.select_list == []:
            self.select_list = list(np.arange(self.n_images))

        self.skip_frames = None
        if os.path.exists(os.path.join(root_dir, 'skip_frames.txt')):
            self.skip_frames = read_txt(os.path.join(root_dir, 'skip_frames.txt'))

    def load_dir(self, dir, select_list=[]):
        img_arr = []
        name_list = os.listdir(dir)
        name_list.sort()
        self.n_images_all = len(name_list)
        print('Check image order:', name_list[:3])
        # only load selected images
        if select_list != []:
            select_name_list = [name_list[idx] for idx in select_list]
        else:
            select_name_list = name_list
        # load images
        for name in select_name_list:
            img = load_image(os.path.join(dir, name))
            img_arr.append(img)
        return img_arr
    
    def get_camera_details(self, camera_path, select_list):
        cam, img, _ = read_model(camera_path)
        # camera info
        self.camera_info = CameraInfo()
        self.camera_info.h = cam[1].height
        self.camera_info.w = cam[1].width
        self.camera_info.fx = cam[1].params[0]
        self.camera_info.fy = cam[1].params[0]
        self.camera_info.cx = cam[1].params[1]
        self.camera_info.cy = cam[1].params[2]
        print(self.camera_info.fx, self.camera_info.cx, self.camera_info.cy)
        print(len(img.keys()))

        # cam pose (world to camera)
        colmap_to_gl = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ])
        self.camera_poses = []
        for i in range(self.n_images_all):
            rmat = img[i+1].qvec2rotmat()
            tvec = img[i+1].tvec
            mat_3x4 = np.concatenate([rmat, tvec.reshape([3, 1])], axis=1)
            mat_4x4 = np.concatenate([mat_3x4, np.array([[0, 0, 0, 1]])], axis=0)
            self.camera_poses.append(colmap_to_gl @ mat_4x4)
            # self.camera_poses.append(np.linalg.inv(mat_4x4))
        if select_list != []:
            self.camera_poses = [self.camera_poses[i] for i in select_list]

    def __getitem__(self, index):
        image = to_tensor(self.images[index])
        iris_mask = to_tensor(self.iris_masks[index])
        eye_mask = to_tensor(self.eye_masks[index])
        camera_pose = to_tensor(self.camera_poses[index])

        return image, iris_mask, eye_mask, camera_pose, index

    def __len__(self):
        return self.n_images


def get_colmap_loader(config):
    dataset = ColmapDataset(config['data'], config['data']['load_img'])
    train_dataloader = DataLoader(dataset, 
                            batch_size=config['optim']['batch'],
                            shuffle=False, drop_last=False)
    test_dataloader = DataLoader(dataset, 
                            batch_size=1,
                            shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader, dataset
