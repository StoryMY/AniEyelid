import os
import numpy as np
import torch
import cv2
import imageio
from PIL import Image
import trimesh
import logging
import time
import yaml
import nvdiffrec_render.util

class Timer():
    def __init__(self, name):
        self.name = name
        self.st = time.time()
    
    def report(self):
        now = time.time()
        print(self.name, now - self.st)

class LossLogger():
    def __init__(self, out_dir, log_name):
        # init file logger
        self.logger = logging.getLogger('Logger' + log_name)
        self.logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(os.path.join(out_dir, log_name + '.log'))
        f_handler.setLevel(logging.INFO)
        # f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(f_handler)

        # init log dict
        self.log_item = {}
    
    def init(self, name):
        self.log_item[name] = {}
        self.log_item[name]['num'] = 0
        self.log_item[name]['loss'] = 0
    
    def update(self, name, loss, num):
        self.log_item[name]['num'] += num
        self.log_item[name]['loss'] += loss * num

    def report(self, name, epoch):
        err = self.log_item[name]['loss'] / self.log_item[name]['num']
        self.logger.info('[Epoch %d] %s: %f' % (epoch, name, err))
        print('[Epoch %d] %s: %f' % (epoch, name, err))
    
    def info(self, name, epoch, val):
        self.logger.info('[Epoch %d] %s: %s' % (epoch, name, str(val)))
        print('[Epoch %d] %s: %s' % (epoch, name, str(val)))

def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def smart_mkdir(dir, name):
    log_dir = os.path.join(dir, name)
    samp_dir = os.path.join(log_dir, 'sample')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(samp_dir, exist_ok=True)

    return (log_dir, samp_dir)

def to_tensor(np_arr, dtype=np.float32):
    return torch.from_numpy(np_arr.astype(dtype))

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        assert isinstance(x, list)
        ret = []
        for item in x:
            item_np = item.detach().cpu().numpy()
            ret.append(item_np)

        return ret

def mask_to_xy(mask):
    assert len(mask.shape) == 3, mask.shape # [H, W, 1]
    assert mask.shape[2] == 1, mask.shape
    x_range = torch.arange(mask.shape[1])
    y_range = torch.arange(mask.shape[0])

    y, x = torch.meshgrid(y_range, x_range, indexing="ij")
    yx = torch.stack((y, x), dim=2).reshape(mask.shape[0], mask.shape[1], 2)
    yx_t = yx.type(torch.float32).to(mask.device)
    xy_t = torch.flip(yx_t, dims=[2])

    select = mask.ge(0.5).squeeze(-1)
    mask_xy = xy_t[select]

    return mask_xy

def center_crop(img, w=1200, h=1200):
    width = img.shape[1]
    height = img.shape[0]
    cx = width // 2
    cy = height // 2

    l = cx - w // 2
    r = cx + w // 2
    t = cy - h // 2
    b = cy + h // 2

    return img[t:b, l:r], (t, b, l, r)

def tonemap(hdr, u=10):
    return np.log(1 + u * hdr) / np.log(1 + u)

def load_envimage(path):
    latlong_map = Image.open(path)
    latlong_map = np.array(latlong_map) / 255
    assert latlong_map.shape[2] == 3, latlong_map.shape
    # latlong_map = to_tensor(cv2.flip(latlong_map, 0)).to('cuda')
    latlong_map = to_tensor(latlong_map).to('cuda')
    cube_maps = nvdiffrec_render.util.latlong_to_cubemap(latlong_map, [512, 512])
    return cube_maps

def save_envimage(save_path, cube_maps, save_cubes=False):
    if save_cubes:
        for i in range(6):
            cube = cube_maps[i]
            save_image(save_path[:-4] + '_%d.png' % i, cube.detach().cpu().numpy(), flip=False)
    latlong_map = nvdiffrec_render.util.cubemap_to_latlong(cube_maps, [512, 1024])
    # save_image(save_path, cv2.flip(latlong_map.detach().cpu().numpy(), 0))   # rotate 180 degree
    save_image(save_path, latlong_map.detach().cpu().numpy(), flip=False)

def load_envimage_hdr(path):
    if path[-3:] != 'hdr':
        print('Not HDR!')
        return load_envimage(path)
    assert path[-3:] == 'hdr'
    latlong_map = imageio.imread(path, 'hdr')
    latlong_map = np.array(latlong_map)
    # latlong_map = to_tensor(cv2.flip(latlong_map, 0)).to('cuda')
    latlong_map = to_tensor(latlong_map).to('cuda')
    cube_maps = nvdiffrec_render.util.latlong_to_cubemap(latlong_map, [512, 512])
    return cube_maps

def save_envimage_hdr(save_path, cube_maps, save_cubes=False):
    assert save_path[-3:] == 'hdr'
    if save_cubes:
        for i in range(6):
            cube = cube_maps[i]
            save_image(save_path[:-4] + '_%d.png' % i, cube.detach().cpu().numpy(), flip=False)
    latlong_map = nvdiffrec_render.util.cubemap_to_latlong(cube_maps, [512, 1024])
    # latlong_map_save = cv2.flip(latlong_map.detach().cpu().numpy(), 0)
    latlong_map_save = latlong_map.detach().cpu().numpy()
    imageio.imwrite(save_path, latlong_map_save, 'hdr')
    save_image(save_path[:-4] + '_rgb.png', tonemap(latlong_map_save), flip=False)   # save png env

def load_image(path, flip=True):
    img = Image.open(path)
    img_arr = np.array(img) / 255
    if flip:
        img_arr = cv2.flip(img_arr, 0)  # inverse up/down
    if len(img_arr.shape) == 2:
        img_arr = img_arr[..., np.newaxis]
    return img_arr

def save_image(save_path, img_arr, norm=True, flip=True):
    """
        save_path: str
        img_arr: numpy array
    """
    if img_arr.shape[2] == 1:
        img_arr = img_arr[..., 0]
    if norm:
        img_arr = img_arr * 255
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    if flip:
        img_arr = cv2.flip(img_arr, 0)  # inverse up/down
    img = Image.fromarray(img_arr)
    img.save(save_path)

def save_video(save_path, img_dir, fps=30):
    writer = imageio.get_writer(save_path, fps=fps)
    img_list = os.listdir(img_dir)
    img_list.sort()
    for img_name in img_list:
        img = imageio.imread(os.path.join(img_dir, img_name))
        writer.append_data(img)
    writer.close()

def translate_to_matrix(vec):
    rot_mat = torch.eye(3, 3, dtype=torch.float32).to(vec.device)
    w = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(vec.device)
    mat_3x4 = torch.cat([rot_mat, vec.reshape([3, 1])], dim=1)
    mat_4x4 = torch.cat([mat_3x4, w], dim=0)

    return mat_4x4

def _axis_angle_rotation(axis: str, angle: torch.Tensor):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 4, 4).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, 
                zero, cos, -sin, zero,
                zero, sin, cos, zero,
                zero, zero, zero, one)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero,
                zero, one, zero, zero,
                -sin, zero, cos, zero,
                zero, zero, zero, one)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, zero,
                sin, cos, zero, zero,
                zero, zero, one, zero,
                zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (4, 4))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 4, 4).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] < 2:
        raise ValueError("Invalid input euler angles.")
    if len(convention) < 2:
        raise ValueError("Convention at least have 2 letters.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    if len(matrices) == 3:
        return torch.mm(torch.mm(matrices[2], matrices[1]), matrices[0])
    return torch.mm(matrices[1], matrices[0])

def gen_gaze(pitch_max, pitch_num, yaw_max, yaw_num):
    pitch = np.linspace(-1, 1, pitch_num) * pitch_max
    yaw = np.linspace(-1, 1, yaw_num) * yaw_max

    col, row = np.meshgrid(yaw, pitch)
    gaze = np.stack([row, col], axis=2).reshape([-1, 2])

    assert gaze.shape[0] == pitch_num * yaw_num

    return gaze

def gen_circle_gaze(pitch_max, yaw_max, gaze_num):
    pitch = np.linspace(-1, 1, gaze_num) * pitch_max
    yaw = np.linspace(-1, 1, gaze_num) * yaw_max
    pitch_inv = pitch[::-1]
    yaw_inv = yaw[::-1]
    pitch = np.concatenate([pitch[:-1], pitch_inv[:-1]])
    yaw = np.concatenate([yaw[:-1], yaw_inv[:-1]])

    gaze = []
    p_st = gaze_num // 2
    y_st = 0
    for i in range(2*gaze_num-2):
        p_idx = (p_st + i) % (2*gaze_num-2)
        y_idx = (y_st + i) % (2*gaze_num-2)

        temp = [pitch[p_idx], yaw[y_idx]]
        gaze.append(temp)

    return np.array(gaze)

def generate_cornea(z_num, z_max, theta_num, e=0.5, curv_r=7.8):
    """
        :param z_num: layer number of z
        :param z_max: cornea height
        :param theta_num: number of samples (360 degrees)
        :param e: eccentricity
        :param curv_r: radius of curvanture
    """
    theta_gap = 2 * np.pi / theta_num
    p = 1 - e * e

    z_arr = np.arange(z_num + 1) / z_num
    z_arr = z_arr ** 2 * z_max

    # draw a circle
    points_arr = []
    for z in z_arr:
        for theta_idx in range(theta_num):
            theta = theta_idx * theta_gap
            sqrt_v = np.sqrt(-p * z * z + 2 * curv_r * z)
            x = sqrt_v * np.cos(theta)
            y = sqrt_v * np.sin(theta)
            points_arr.append([x, y, z])
    
    points = np.array(points_arr)
    cloud = trimesh.PointCloud(points)
    # cloud.export('cornea_pointcloud.obj')
    mesh = cloud.convex_hull
    mesh.export('cornea_z%d_t%d.obj' % (z_num, theta_num))
    

if __name__ == '__main__':
    # generate_cornea(10, 3.5, 24)    # cornea_equation_v2.obj
    # generate_cornea(5, 3.5, 24)

    mask = torch.zeros([4, 2, 1], dtype=torch.float32)
    mask[0, 1] = 1
    mask[1, 1] = 1
    mask[3, 1] = 1
    mask_xy = mask_to_xy(mask)
    print(mask_xy)
