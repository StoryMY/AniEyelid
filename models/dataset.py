import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import logging
from tqdm import tqdm

from .LieAlgebra import se3

def read_txt(path):
    ret = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ret.append(int(line.strip()))
    return ret

# This function is based upon IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_eyelid_region(mask_arr):
    print('get eyelid surround masks..')
    eyelid_surround_mask = []
    for mask in tqdm(mask_arr):
        H, W = mask.shape[:2]
        y, x = np.where(mask[:, :, 0] == 1.0)
        try:
            x_min, x_max = max(np.min(x) - 100, 0), min(np.max(x) + 100, W-1)
            y_min, y_max = max(np.min(y) - 100, 0), min(np.max(y) + 100, H-1)
            lid_region = np.zeros([H, W, 3], dtype=np.float32)
            lid_region[y_min:y_max, x_min:x_max] = 1.0
        except:
            lid_region = eyelid_surround_mask[-1].copy()

        eyelid_surround_mask.append(lid_region)
    
    return np.stack(eyelid_surround_mask)


def imread_resize(path, ds, binary=False, cvt=False):
    img = cv.imread(path)
    if cvt:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if ds > 1:
        w = int(img.shape[1] / ds + 0.5)
        h = int(img.shape[0] / ds + 0.5)
        img = cv.resize(img, (w, h))
        if binary:
            img[img > 128] = 255
            img[img < 128] = 0
    return img


def random_sample(mask, batch_size, device):
    y, x = np.where(mask[:, :, 0] == 1)
    idx = np.random.randint(0, y.shape[0], batch_size)
    pixel_x = torch.from_numpy(x[idx]).to(device)
    pixel_y = torch.from_numpy(y[idx]).to(device)

    return pixel_x, pixel_y


class MiniDataset:
    def __init__(self, conf):
        logging.info('Load data: Begin Mini')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dtype = torch.get_default_dtype()

        # Camera
        self.is_monocular = conf.get_bool('is_monocular')
        self.camera_trainable = conf.get_bool('camera_trainable')

        self.data_dir = conf.get_string('data_dir')
        self.mask_dir = conf.get_string('mask_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.img_downscale = conf.get_float('img_downscale', default=1.0)

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        if len(self.images_lis) == 0:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        self.n_images = len(self.images_lis)
        image_template = imread_resize(self.images_lis[0], self.img_downscale) / 256.0
        self.H, self.W = image_template.shape[0], image_template.shape[1]

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        poses_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # img scale
            intrinsics[0, 0] /= self.img_downscale
            intrinsics[1, 1] /= self.img_downscale
            intrinsics[0, 2] /= self.img_downscale
            intrinsics[1, 2] /= self.img_downscale

            intrinsics_all.append(torch.from_numpy(intrinsics).to(self.dtype))
            poses_all.append(torch.from_numpy(pose).to(self.dtype)) # the inverse of extrinsic matrix
        intrinsics_all = torch.stack(intrinsics_all).to(self.device) # [n_images, 4, 4]
        poses_all = torch.stack(poses_all).to(self.device) # [n_images, 4, 4]

        # Camera
        if self.is_monocular:
            self.intrinsics_paras = torch.stack((intrinsics_all[:1,0,0], intrinsics_all[:1,1,1], \
                                                intrinsics_all[:1,0,2], intrinsics_all[:1,1,2]),
                                                    dim=1) # [1, 4]: (fx, fy, cx, cy)
        else:
            self.intrinsics_paras = torch.stack((intrinsics_all[:,0,0], intrinsics_all[:,1,1], \
                                                intrinsics_all[:,0,2], intrinsics_all[:,1,2]),
                                                    dim=1) # [n_images, 4]: (fx, fy, cx, cy)
        self.poses_paras = se3.log(poses_all) # [n_images, 6]
        if self.camera_trainable:
            self.intrinsics_paras.requires_grad_()
            self.poses_paras.requires_grad_()
        else:
            self.static_paras_to_mat()

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        # Gaze and Eyeball
        self.use_gaze = conf.get_bool('use_gaze', default=False)
        self.use_split = conf.get_bool('use_split', default=False)
        self.use_eyeball = conf.get_bool('use_eyeball', default=False)
        self.use_eyebbox = conf.get_bool('use_eyebbox', default=True)
        if self.use_gaze:
            # Gaze
            self.gaze_file_name = conf.get_string('gaze_name')
            if self.use_split:
                gaze_lis_os = np.load(os.path.join(self.data_dir, self.gaze_file_name))['gaze']
                gaze_lis_od = np.load(os.path.join(self.data_dir, self.gaze_file_name))['gaze2']
                self.gaze_lis = np.concatenate([np.stack(gaze_lis_os), np.stack(gaze_lis_od)], axis=-1).astype(np.float32)
                self.gaze_lis = torch.from_numpy(self.gaze_lis).to(self.dtype).to(self.device)
                assert self.gaze_lis.shape[0] == self.n_images, self.gaze_lis.shape
                assert self.gaze_lis.shape[1] == 4, self.gaze_lis.shape
            else:
                gaze_lis = np.load(os.path.join(self.data_dir, self.gaze_file_name))['gaze']
                self.gaze_lis = np.stack(gaze_lis).astype(np.float32)
                self.gaze_lis = torch.from_numpy(self.gaze_lis).to(self.dtype).to(self.device)
                assert self.gaze_lis.shape[0] == self.n_images, self.gaze_lis.shape
                assert self.gaze_lis.shape[1] == 2, self.gaze_lis.shape

            self.skip_frames = []
            if os.path.exists(os.path.join(self.data_dir, 'skip_frames.txt')):
                self.skip_frames = read_txt(os.path.join(self.data_dir, 'skip_frames.txt'))
                self.gaze_lis[self.skip_frames] = torch.zeros([self.gaze_lis.shape[1]])
                print('[Set Gaze 0]:', self.skip_frames)
        
        self.inside_info = None
        self.eye_bbox = None
        if self.use_eyebbox:
            # Eyeball Sample Points
            self.esp_file_name = conf.get_string('eyeball_name')
            self.esp_mode = conf.get_string('eyeball_mode', default='both')
            assert self.esp_mode in ['both', 'up', 'down'], self.esp_mode

            data_root = np.load(os.path.join(self.data_dir, self.esp_file_name), allow_pickle=True)
            esp_xyz_lis = data_root['xyz']
            esp_nor_lis = data_root['nor']

            if 'xyz_half' in data_root.keys():
                esp_xyz_half_lis = data_root['xyz_half']
                self.eyeball_scale = conf.get_float('eyeball_scale', default=1.0)
                print('[use eyeball] load eyeball scale:', self.eyeball_scale)
                esp_xyz_lis = esp_xyz_lis + (esp_xyz_half_lis - esp_xyz_lis) / 0.5 * (1 - self.eyeball_scale)
            else:
                self.eyeball_scale = 1.0
                print('[use eyeball] no eyeball scale!')

            assert len(esp_xyz_lis) == self.n_images, len(esp_xyz_lis)
            # convert image-space to bbox-space
            self.esp_xyz_bbox_lis = []
            self.esp_nor_bbox_lis = []
            select_angle = np.pi/6
            for img_idx in range(self.n_images):
                if self.esp_mode == 'both':
                    select = esp_nor_lis[img_idx][:, 2] * -1 > np.cos(select_angle)                                          # both
                elif self.esp_mode == 'down':
                    select = (esp_nor_lis[img_idx][:, 2] * -1 > np.cos(select_angle)) & (esp_nor_lis[img_idx][:, 1] > 0)     # down
                else:
                    select = (esp_nor_lis[img_idx][:, 2] * -1 > np.cos(select_angle)) & (esp_nor_lis[img_idx][:, 1] < 0)     # up

                xyz = esp_xyz_lis[img_idx][select]
                xyz_half = esp_xyz_half_lis[img_idx][select]
                nor = esp_nor_lis[img_idx][select]

                xyz_scale = xyz / self.scale_mats_np[0][0, 0]
                xyz_scale = np.concatenate([xyz_scale, np.ones([xyz_scale.shape[0], 1])], axis=1)
                xyz_bbox = np.matmul(xyz_scale, self.poses_all[img_idx].cpu().numpy().T)[:, :3]
                
                if img_idx == 0:
                    xyz_half_scale = xyz_half / self.scale_mats_np[0][0, 0]
                    xyz_half_scale = np.concatenate([xyz_half_scale, np.ones([xyz_half_scale.shape[0], 1])], axis=1)
                    xyz_half_bbox = np.matmul(xyz_half_scale, self.poses_all[img_idx].cpu().numpy().T)[:, :3]
                    
                    radius_r = np.linalg.norm(xyz_half_bbox[0] - xyz_bbox[0], ord=2, axis=-1) * 2       # OS, image right eye
                    center_r = (xyz_half_bbox[0] - xyz_bbox[0]) * 2 + xyz_bbox[0]
                    radius_l = np.linalg.norm(xyz_half_bbox[-1] - xyz_bbox[-1], ord=2, axis=-1) * 2     # OD, image left eye
                    center_l = (xyz_half_bbox[-1] - xyz_bbox[-1]) * 2 + xyz_bbox[-1]
                    distance = np.linalg.norm(center_r - center_l, ord=2, axis=-1)
                    center = (center_r + center_l) * 0.5
                    # Eye bbox
                    self.eye_bbox_min = np.array([*(center - distance*np.array([1.0, 0.6, 0.5]))])
                    self.eye_bbox_max = np.array([*(center + distance*np.array([1.0, 0.4, 0.5]))])
                    self.eye_bbox = np.concatenate([self.eye_bbox_min, self.eye_bbox_max], axis=-1)
                    # OS bbox
                    os_temp = (center_r - center) * 0.2 + center
                    self.os_bbox_min = np.array([os_temp[0], self.eye_bbox_min[1], self.eye_bbox_min[2]])
                    self.os_bbox_max = self.eye_bbox_max
                    self.os_bbox = np.concatenate([self.os_bbox_min, self.os_bbox_max], axis=-1)
                    # OD bbox
                    od_temp = (center_l - center) * 0.2 + center
                    self.od_bbox_min = self.eye_bbox_min
                    self.od_bbox_max = np.array([od_temp[0], self.eye_bbox_max[1], self.eye_bbox_max[2]])
                    self.od_bbox = np.concatenate([self.od_bbox_min, self.od_bbox_max], axis=-1)

                    x_axis_min = center_l[0] - radius_l
                    x_axis_max = center_r[0] + radius_r
                    z_axis_max = center[2] + distance * 0.25
                    self.inside_info = {'x_min': x_axis_min, 'x_max': x_axis_max, 'z_max': z_axis_max}

                nor = np.concatenate([nor, np.zeros([nor.shape[0], 1])], axis=1)
                nor_bbox = np.matmul(nor, self.poses_all[img_idx].cpu().numpy().T)[:, :3]
                self.esp_xyz_bbox_lis.append(xyz_bbox)
                self.esp_nor_bbox_lis.append(nor_bbox)

        print('inside_info:', self.inside_info)
        logging.info('Load data: End Mini')

    # Camera
    def static_paras_to_mat(self):
        fx, fy, cx, cy = self.intrinsics_paras[:,0], self.intrinsics_paras[:,1],\
                            self.intrinsics_paras[:,2], self.intrinsics_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_all_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        self.intrinsics_all_inv = torch.cat((torch.cat(
                                (intrinsics_all_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        self.poses_all = se3.exp(self.poses_paras)

    def get_gaze(self, idx):
        return self.gaze_lis[idx]
    
    def get_esp_xyz_nor(self, idx):
        xyz = self.esp_xyz_bbox_lis[idx]
        xyz = torch.from_numpy(xyz).to(self.dtype).to(self.device)
        nor = self.esp_nor_bbox_lis[idx]
        nor = torch.from_numpy(nor).to(self.dtype).to(self.device)
        return xyz, nor

class Dataset(MiniDataset):
    def __init__(self, conf):
        super(Dataset, self).__init__(conf)
        logging.info('Load data: Begin')

        self.use_normal = conf.get_bool('use_normal', default=False)

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        if len(self.images_lis) == 0:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([imread_resize(im_name, self.img_downscale) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, self.mask_dir + '/*.jpg')))
        if len(self.masks_lis) == 0:
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, self.mask_dir + '/*.png')))
        self.masks_np = np.stack([imread_resize(im_name, self.img_downscale, binary=True) for im_name in self.masks_lis]) / 255.0

        self.fulmasks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        if len(self.fulmasks_lis) == 0:
            self.fulmasks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.fullmasks_np = np.stack([imread_resize(im_name, self.img_downscale, binary=True) for im_name in self.fulmasks_lis]) / 255.0
        self.eyemasks_np = self.fullmasks_np - self.masks_np
        self.eyemasks_np[self.eyemasks_np > 0.5] = 1.0
        self.eyemasks_np[self.eyemasks_np < 0.5] = 0.0

        # eyelid weight masks
        ewm_path = os.path.join(self.data_dir, 'eyelid_surround_mask.npy')
        if os.path.exists(ewm_path):
            self.eyelid_surround_mask = np.load(ewm_path, allow_pickle=True)
            print('load eyelid surround masks from:', ewm_path)
        else:
            self.eyelid_surround_mask = get_eyelid_region(self.eyemasks_np)
            np.save(ewm_path, self.eyelid_surround_mask)
            print('save eyelid surround masks to:', ewm_path)
        assert self.eyelid_surround_mask.shape == self.masks_np.shape
        print('eyelid surround:', np.max(self.eyelid_surround_mask))
        print('eyelid surround:', self.eyelid_surround_mask.shape)

        if self.use_normal:
            self.normal_lis = sorted(glob(os.path.join(self.data_dir, 'normal/*.png')))
            self.normals_np = np.stack([imread_resize(im_name, self.img_downscale, cvt=True) for im_name in self.normal_lis]) / 127.5 - 1
            # invert 
            self.normals_np[:, :, :, 1] = -self.normals_np[:, :, :, 1]
            self.normals_np[:, :, :, 2] = -self.normals_np[:, :, :, 2]
            self.normals  = torch.from_numpy(self.normals_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 1]

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 1]
        self.eyemasks  = torch.from_numpy(self.eyemasks_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 1]
        self.H, self.W = self.images.shape[1], self.images.shape[2]

        logging.info('Load data: End')


    def dynamic_paras_to_mat(self, img_idx, add_depth=False):
        if self.is_monocular:
            intrinsic_paras = self.intrinsics_paras[:1, :]
        else:
            intrinsic_paras = self.intrinsics_paras[img_idx:(img_idx+1), :]
        fx, fy, cx, cy = intrinsic_paras[:,0], intrinsic_paras[:,1], intrinsic_paras[:,2], intrinsic_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        intrinsic_inv = torch.cat((torch.cat(
                                (intrinsics_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        pose_paras = self.poses_paras[img_idx:(img_idx+1), :]
        pose = se3.exp(pose_paras)

        return intrinsic_inv.squeeze(), pose.squeeze()


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        query_mask = torch.stack([pixels_x, pixels_y], dim=-1).long()
        eyemask = self.eyemasks[img_idx].to(self.device)[(query_mask[..., 1], query_mask[..., 0])]
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), eyemask[..., :1].transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size, sample_more_eye=False):
        """
        Generate random rays at world space from one camera.
        """
        if not sample_more_eye:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        else:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size // 2])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size // 2])
            pixels_x_eye, pixels_y_eye = random_sample(self.eyelid_surround_mask[img_idx], batch_size // 2, self.device)
            pixels_x = torch.cat([pixels_x_eye, pixels_x])
            pixels_y = torch.cat([pixels_y_eye, pixels_y])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 1
        eyemask = self.eyemasks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 1
        # lid_weight = self.eyelid_surround_mask[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        if self.use_normal:
            normal = self.normals[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
            return torch.cat([rays_o, rays_v, color, mask[:, :1], eyemask[:, :1], normal], dim=-1) # batch_size, 14
        return torch.cat([rays_o, rays_v, color, mask[:, :1], eyemask[:, :1]], dim=-1) # batch_size, 11


    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


    def get_image_size(self):
        return self.H, self.W
