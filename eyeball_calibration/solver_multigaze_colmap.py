import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import trimesh
import cv2

import nvdiffrast.torch as dr

import rast_util
import utils
from parametric_eyeball import ParametricEyeball
from render import SimpleRender, CameraInfo

def save_pcd_camera(pcd, cam_pose, save_path):
    gl_to_cv = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
    verts_w = pcd.vertices
    verts_wh = np.concatenate([verts_w, np.ones([verts_w.shape[0], 1])], axis=-1)
    verts_c = verts_wh @ (gl_to_cv @ cam_pose).T
    pcd_c = trimesh.PointCloud(verts_c[:, :3])
    pcd_c.export(save_path)

class EyeballParameter():
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.n_images = self.dataset.n_images
        self.device = device

    def init_params(self, data_dir, optim_config, is_first=True, save_path=None):
        if 'init_gaze' in optim_config.keys():
            gaze = []
            for i in range(self.n_images):
                gaze += [np.array(optim_config['init_gaze'])]
            self.gaze = utils.to_tensor(np.stack(gaze)).to(self.device)
            print('gaze shape', self.gaze.shape)
        
        if 'init_tohead' in optim_config.keys():
            self.tohead = utils.to_tensor(np.array(optim_config['init_tohead'])).to(self.device)

        if 'init_translate' in optim_config.keys():
            if 'init_npz' in optim_config.keys():
                param_npz = np.load(optim_config['init_npz'])
                translate_key = 'translate' if is_first else 'translate2'
                self.translate = utils.to_tensor(np.array(param_npz[translate_key])).to(self.device)
            else: 
                pcd = trimesh.load_mesh(os.path.join(data_dir, optim_config['init_translate']))
                vertices = pcd.vertices
                bbox_max = np.max(vertices, axis=0)
                bbox_min = np.min(vertices, axis=0)
                self.translate = (bbox_max + bbox_min) * 0.5
                if 'ft_translate' in optim_config.keys():
                    print('ft_translate:', optim_config['ft_translate'])
                    self.translate = self.translate + np.array(optim_config['ft_translate'])
                self.translate = utils.to_tensor(self.translate).to(self.device)
                if save_path is not None:
                    save_pcd_camera(pcd, self.dataset.camera_poses[0], os.path.join(save_path, 'pcd_0.ply'))

        if 'init_scale' in optim_config.keys():
            if 'init_npz' in optim_config.keys():
                param_npz = np.load(optim_config['init_npz'])
                scale_key = 'scale' if is_first else 'scale2'
                self.scale = utils.to_tensor(np.array(param_npz[scale_key])).to(self.device)
            else: 
                self.scale = utils.to_tensor(np.array(optim_config['init_scale'])).to(self.device)
        else:
            self.scale = utils.to_tensor(np.array(1.0)).to(self.device)
        print('translate:', self.translate)
        print('scale:', self.scale)


class ColmapGazeSolver():
    def __init__(self, config, out_info, dataset) -> None:
        self.config = config
        self.log_dir, self.samp_dir = out_info
        self.device = config['device']

        self.p_eyeball = ParametricEyeball(config['model']['dir'], self.device)

        self.dataset = dataset
        self.split = config['data']['split']    # assume first 'split' frames share the same gaze
        self.camera = dataset.camera_info
        self.n_images = dataset.n_images
        self.renderer = SimpleRender(self.camera, self.device)

        self.init_params()

    def init_params(self):
        self.eb_params = EyeballParameter(self.dataset, self.device)
        self.eb_params.init_params(self.config['data']['root_dir'], self.config['optim'], save_path=self.log_dir)
        
        self.env = torch.ones([6, 512, 512, 3], dtype=torch.float32).to(self.device) * .5
    
    def get_gaze(self, idx):
        if idx < 0:
            return self.eb_params.gaze[0]
        if self.dataset.select_list[idx] < self.split:
            return self.eb_params.gaze[0]
        return self.eb_params.gaze[idx]

    def set_gaze(self):
        for idx in range(self.eb_params.gaze.shape[0]):
            if self.dataset.select_list[idx] < self.split:
                self.eb_params.gaze[idx] = self.eb_params.gaze[0].clone()

    def get_gaze_reg(self):
        gaze_reg_list = []
        for idx in range(self.eb_params.gaze.shape[0]):
            gaze_reg = 2 * self.get_gaze(idx-1) - self.get_gaze(idx-2)
            gaze_reg_list.append(gaze_reg)
        return gaze_reg_list

    def pose_assemble(self, gaze, cam_pose, eb_params):
        tohead_mtx = utils.euler_angles_to_matrix(eb_params.tohead, "XYZ")   # maybe need add translate
        gaze_mtx = utils.euler_angles_to_matrix(gaze, "XY")
        trans_mtx = utils.translate_to_matrix(eb_params.translate)

        # return torch.mm(cam_pose, torch.mm(tohead_mtx, torch.mm(trans_mtx, gaze_mtx)))  # 4x4
        return torch.mm(cam_pose, torch.mm(trans_mtx, torch.mm(tohead_mtx, gaze_mtx)))  # 4x4

    def solve_proj(self, train_dataloader, test_dataloader):
        max_iter = self.config['optim']['max_iter'][0]
        self.eb_params.translate.requires_grad_(True)
        self.eb_params.scale.requires_grad_(True)
        optimizer = torch.optim.Adam([self.eb_params.translate, self.eb_params.scale], lr=self.config['optim']['lr_base'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: self.config['optim']['lr_ramp']**(float(x)/float(max_iter)))
        
        glctx = dr.RasterizeCudaContext()

        eyeball_info = self.p_eyeball.get_nvdiffrast_info()
        eyeball_info.to_tensor(self.device)
        cornea_info = self.p_eyeball.get_nvdiffrast_info_cornea()
        cornea_info.to_tensor(self.device)
        iris_proxy_info = self.p_eyeball.get_nvdiffrast_info_iris_proxy()
        iris_proxy_info.to_tensor(self.device)

        log_interval = self.config['optim']['log_interval']
        imgsave_interval = self.config['optim']['imgsave_interval']
        logger = utils.LossLogger(self.log_dir, 'proj')
        for epoch in range(max_iter + 1):
            logger.init('loss')
            for step, (_, iris_mask, eye_mask, cam_pose, gaze_idx) in enumerate(train_dataloader):
                iris_mask = iris_mask.to(self.device)
                eye_mask = eye_mask.to(self.device)
                cam_pose = cam_pose.to(self.device)
                eyeball_info.set_scale(self.eb_params.scale)
                cornea_info.set_scale(self.eb_params.scale)
                iris_proxy_info.set_scale(self.eb_params.scale)

                batch_size = gaze_idx.shape[0]
                # process batch_size
                proj_center = []
                proj_radius = []
                gt_center = []
                gt_radius = []
                for bid in range(batch_size):
                    mtx = self.pose_assemble(self.get_gaze(gaze_idx[bid]), cam_pose[bid], self.eb_params)

                    pos_screen = self.renderer.project_points(mtx, iris_proxy_info.pos * iris_proxy_info.scale)
                    proj_center_item = torch.mean(pos_screen[:, :2], dim=0) / iris_mask.shape[2]
                    proj_radius_item = torch.max(torch.max(pos_screen, dim=0)[0] - torch.min(pos_screen, dim=0)[0]) / iris_mask.shape[2]
                    
                    mask_xy = utils.mask_to_xy(iris_mask[bid])
                    gt_center_item = torch.mean(mask_xy, dim=0) / iris_mask.shape[2]
                    gt_radius_item = torch.max(torch.max(mask_xy, dim=0)[0] - torch.min(mask_xy, dim=0)[0]) / iris_mask.shape[2]

                    proj_center.append(proj_center_item)
                    proj_radius.append(proj_radius_item)
                    gt_center.append(gt_center_item)
                    gt_radius.append(gt_radius_item)
                proj_center = torch.stack(proj_center)
                proj_radius = torch.stack(proj_radius)
                gt_center = torch.stack(gt_center)
                gt_radius = torch.stack(gt_radius)

                ## assume batch_size = 1
                # mtx = self.pose_assemble(self.get_gaze(gaze_idx[0]), cam_pose[0], self.eb_params)
            
                # pos_screen = self.renderer.project_points(mtx, iris_proxy_info.pos * iris_proxy_info.scale)
                # proj_center = torch.mean(pos_screen[:, :2], dim=0) / iris_mask.shape[2]
                # proj_radius = torch.max(torch.max(pos_screen, dim=0)[0] - torch.min(pos_screen, dim=0)[0]) / iris_mask.shape[2]
                
                # mask_xy = utils.mask_to_xy(iris_mask[0])
                # gt_center = torch.mean(mask_xy, dim=0) / iris_mask.shape[2]
                # gt_radius = torch.max(torch.max(mask_xy, dim=0)[0] - torch.min(mask_xy, dim=0)[0]) / iris_mask.shape[2]

                location_loss = torch.mean(torch.abs(gt_center - proj_center))
                radius_loss = torch.mean(torch.abs(gt_radius - proj_radius))
                loss = location_loss + radius_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.update('loss', loss.item(), batch_size)
            
            scheduler.step()

            if log_interval and (epoch % log_interval == 0):
                logger.report('loss', epoch)
                logger.info('trans', epoch, self.eb_params.translate.detach().cpu().numpy())
                logger.info('proj_center', epoch, proj_center.detach().cpu().numpy() * iris_mask.shape[2])
                logger.info('gt_center', epoch, gt_center.detach().cpu().numpy() * iris_mask.shape[2])
                logger.info('proj_radius', epoch, proj_radius.detach().cpu().numpy())
                logger.info('gt_radius', epoch, gt_radius.detach().cpu().numpy())
                logger.info('scale', epoch, self.eb_params.scale.detach().cpu().numpy())

            if imgsave_interval and (epoch % imgsave_interval == 0):
                with torch.no_grad():
                    default_env = utils.load_envimage_hdr(self.config['data']['env'])
                    tex = self.p_eyeball.get_mean_texture_torch()
                    for step, (img, iris_mask, eye_mask, cam_pose, gaze_idx) in enumerate(test_dataloader):
                        cam_pose = cam_pose.to(self.device)
                        eyeball_info.set_scale(self.eb_params.scale)
                        cornea_info.set_scale(self.eb_params.scale)
                        # mtx = self.pose_assemble(self.gaze[gaze_idx[0]], cam_pose[0], self.eb_params)
                        mtx = self.pose_assemble(self.get_gaze(gaze_idx[0]), cam_pose[0], self.eb_params)

                        color_blend = self.renderer.render_blend(glctx, mtx, eyeball_info, cornea_info, tex, default_env)
                        utils.save_image(os.path.join(self.samp_dir, 'color_dummy_proj_%05d_%d.png' % (epoch, gaze_idx)),
                                        color_blend[0].cpu().numpy())

    def solve_pose(self, train_dataloader, test_dataloader):
        max_iter = self.config['optim']['max_iter'][1]
        self.eb_params.gaze.requires_grad_(True)
        is_init = False
        if 'init_npz' in self.config['optim'].keys():
            is_init = False
            print('[npz init] fix translate and scale!')
            self.eb_params.translate.requires_grad_(False)
            self.eb_params.scale.requires_grad_(False)
            optimizer = torch.optim.Adam([self.eb_params.gaze], lr=self.config['optim']['lr_base'])   # [Important!] Adam will cause different results due to the momentum
        else:
            is_init = True
            self.eb_params.translate.requires_grad_(True)
            self.eb_params.scale.requires_grad_(True)
            optimizer = torch.optim.Adam([self.eb_params.gaze, self.eb_params.translate, self.eb_params.scale], lr=self.config['optim']['lr_base'])
        
        print('is_init:', is_init)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: self.config['optim']['lr_ramp']**(float(x)/float(max_iter)))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: self.config['optim']['lr_ramp']**(float(x)/float(100)))
        
        glctx = dr.RasterizeCudaContext()

        eyeball_info = self.p_eyeball.get_nvdiffrast_info()
        eyeball_info.to_tensor(self.device)
        cornea_info = self.p_eyeball.get_nvdiffrast_info_cornea()
        cornea_info.to_tensor(self.device)
        iris_proxy_info = self.p_eyeball.get_nvdiffrast_info_iris_proxy()
        iris_proxy_info.to_tensor(self.device)

        log_interval = self.config['optim']['log_interval']
        imgsave_interval = self.config['optim']['imgsave_interval']
        logger = utils.LossLogger(self.log_dir, 'pose')
        for epoch in range(max_iter + 1):
            logger.init('loss')
            logger.init('location_loss')
            alpha = epoch / (max_iter + 1)
            # gaze_reg_list = self.get_gaze_reg()
            for step, (_, iris_mask, eye_mask, cam_pose, gaze_idx) in enumerate(train_dataloader):
                iris_mask = iris_mask.to(self.device)
                eye_mask = eye_mask.to(self.device)
                cam_pose = cam_pose.to(self.device)
                eyeball_info.set_scale(self.eb_params.scale)
                cornea_info.set_scale(self.eb_params.scale)
                iris_proxy_info.set_scale(self.eb_params.scale)

                batch_size = gaze_idx.shape[0]
                # process batch_size
                proj_center = []
                proj_radius = []
                gt_center = []
                gt_radius = []
                seg = []
                for bid in range(batch_size):
                    mtx = self.pose_assemble(self.get_gaze(gaze_idx[bid]), cam_pose[bid], self.eb_params)

                    # proj
                    pos_screen = self.renderer.project_points(mtx, iris_proxy_info.pos * iris_proxy_info.scale)
                    proj_center_item = torch.mean(pos_screen[:, :2], dim=0) / iris_mask.shape[2]
                    proj_radius_item = torch.max(torch.max(pos_screen, dim=0)[0] - torch.min(pos_screen, dim=0)[0]) / iris_mask.shape[2]
                    
                    mask_xy = utils.mask_to_xy(iris_mask[bid])
                    gt_center_item = torch.mean(mask_xy, dim=0) / iris_mask.shape[2]
                    gt_radius_item= torch.max(torch.max(mask_xy, dim=0)[0] - torch.min(mask_xy, dim=0)[0]) / iris_mask.shape[2]

                    # seg
                    seg_item = self.renderer.render_segment(glctx, mtx, iris_proxy_info)
                    # mask_xy2 = utils.mask_to_xy(seg_item[0])
                    # proj_center_item2 = torch.mean(mask_xy2, dim=0) / iris_mask.shape[2]
                    # print(proj_center_item, proj_center_item2)
                    # exit(0)

                    proj_center.append(proj_center_item)
                    proj_radius.append(proj_radius_item)
                    gt_center.append(gt_center_item)
                    gt_radius.append(gt_radius_item)
                    seg.append(seg_item)
                proj_center = torch.stack(proj_center)
                proj_radius = torch.stack(proj_radius)
                gt_center = torch.stack(gt_center)
                gt_radius = torch.stack(gt_radius)
                seg = torch.stack(seg)


                # # assume batch_size = 1
                # mtx = self.pose_assemble(self.get_gaze(gaze_idx[0]), cam_pose[0], self.eb_params)

                # # proj
                # pos_screen = self.renderer.project_points(mtx, iris_proxy_info.pos * iris_proxy_info.scale)
                # proj_center = torch.mean(pos_screen[:, :2], dim=0) / iris_mask.shape[2]
                # mask_xy = utils.mask_to_xy(iris_mask[0])
                # gt_center = torch.mean(mask_xy, dim=0) / iris_mask.shape[2]
                    
                # # seg
                # seg = self.renderer.render_segment(glctx, mtx, iris_proxy_info)

                # loss = torch.mean((iris_mask * eye_mask - seg * eye_mask)**2) # L2 pixel loss.
                if not is_init:
                    # seg_loss = torch.mean((iris_mask - seg)**2) # L2 pixel loss.
                    seg_loss = torch.mean(torch.abs(iris_mask - seg)) # L1 pixel loss.
                    location_loss = torch.mean(torch.abs(gt_center - proj_center))
                    radius_loss = torch.mean(torch.abs(gt_radius - proj_radius))
                    # gaze_reg_loss = torch.mean(torch.abs(gaze_reg_list[gaze_idx[0]] - self.get_gaze(gaze_idx[0])))
                    loss = seg_loss * alpha + location_loss * 0.1 * (1-alpha/2) #+ gaze_reg_loss * 0.1
                else:
                    # seg_loss = torch.mean((iris_mask - seg)**2) # L2 pixel loss.
                    seg_loss = torch.mean(torch.abs(iris_mask - seg)) # L1 pixel loss.
                    location_loss = torch.mean(torch.abs(gt_center - proj_center))
                    radius_loss = torch.mean(torch.abs(gt_radius - proj_radius))
                    loss = seg_loss + location_loss * 0.1 + radius_loss * 0.01
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.update('location_loss', location_loss.item(), iris_mask.shape[0])
                logger.update('loss', loss.item(), iris_mask.shape[0])
            
            scheduler.step()
            
            if log_interval and (epoch % log_interval == 0):
                print('-' * 20)
                logger.report('loss', epoch)
                logger.report('location_loss', epoch)
                logger.info('gaze', epoch, self.eb_params.gaze[0].detach().cpu().numpy())
                logger.info('proj_center', epoch, proj_center.detach().cpu().numpy() * iris_mask.shape[2])
                logger.info('gt_center', epoch, gt_center.detach().cpu().numpy() * iris_mask.shape[2])
                logger.info('trans', epoch, self.eb_params.translate.detach().cpu().numpy())
                logger.info('scale', epoch, self.eb_params.scale.detach().cpu().numpy())

        with torch.no_grad():
            if not is_init and self.dataset.skip_frames is not None:
                for fid in self.dataset.skip_frames:
                    self.eb_params.gaze[fid] = self.eb_params.gaze[0].clone()

    def save_results(self, test_dataloader):
        is_init = 'init_npz' not in self.config['optim'].keys()

        # make directory for saving images
        result_dir = os.path.join(self.log_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)

        # broadcast gaze[0] to all static frame
        with torch.no_grad():
            self.set_gaze()

        # save parameters
        np.savez(os.path.join(self.log_dir, 'parameters.npz'),
                gaze=utils.to_numpy(self.eb_params.gaze), 
                translate=utils.to_numpy(self.eb_params.translate),
                scale=utils.to_numpy(self.eb_params.scale))
        with open(os.path.join(self.log_dir, 'gaze_pred.txt'), 'w') as f:
            for g in self.eb_params.gaze:
                f.write('%f %f\n' % (g[0].item(), g[1].item()))

        # save render results
        glctx = dr.RasterizeCudaContext()

        eyeball_info = self.p_eyeball.get_nvdiffrast_info()
        eyeball_info.to_tensor(self.device)
        cornea_info = self.p_eyeball.get_nvdiffrast_info_cornea()
        cornea_info.to_tensor(self.device)
        iris_proxy_info = self.p_eyeball.get_nvdiffrast_info_iris_proxy()
        iris_proxy_info.to_tensor(self.device)
        eyeball_info.set_scale(self.eb_params.scale)
        cornea_info.set_scale(self.eb_params.scale)
        iris_proxy_info.set_scale(self.eb_params.scale)

        # # save mesh
        # import trimesh
        # mtx = self.pose_assemble(self.gaze[0], torch.eye(4, 4, dtype=torch.float32, self.eb_params).to(self.device))
        # pts = eyeball_info.pos
        # pts_trans = self.renderer.transform_vec(mtx, pts)[:, :3]
        # pts_trans = pts_trans.detach().cpu().numpy()
        # faces = eyeball_info.pos_idx.detach().cpu().numpy()
        # mesh = trimesh.Trimesh(pts_trans, faces)
        # mesh.export(os.path.join(self.log_dir, 'mesh.ply'))

        # sample points
        nomask_xyz_arr = []
        nomask_xyz_half_arr = []
        nomask_nor_arr = []
        mask_xyz_arr = []
        mask_xyz_half_arr = []
        mask_nor_arr = []
        with torch.no_grad():
            eye_tex = self.p_eyeball.get_mean_texture_torch()
            for step, (_, _, eye_mask, cam_pose, idx) in enumerate(tqdm(test_dataloader)):
                cam_pose = cam_pose.to(self.device)
                mtx = self.pose_assemble(self.eb_params.gaze[idx[0]], cam_pose[0], self.eb_params)
                # mtx = self.pose_assemble(self.get_gaze(idx[0]), cam_pose[0], self.eb_params)

                sample_points = torch.from_numpy(self.p_eyeball.sample_points.astype(np.float32)).to(self.device)
                sample_normals = torch.from_numpy(self.p_eyeball.sample_normals.astype(np.float32)).to(self.device)
                pos_screen, pos_mtx = self.renderer.project_points(mtx, sample_points * self.eb_params.scale, ret_3d=True)
                pos_screen_half, pos_mtx_half = self.renderer.project_points(mtx, sample_points * self.eb_params.scale * 0.5, ret_3d=True)
                nor_mtx = self.renderer.transform_vec(mtx, sample_normals, v=0)[:, :3].contiguous()
                
                total_points = pos_screen.shape[0]
                nomask_xyz = []
                nomask_xyz_half = []
                nomask_nor = []
                # mask_xyz = []
                # mask_xyz_half = []
                # mask_nor = []
                gl_to_cv = np.array([
                            [1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]
                        ])
                pos_mtx = np.concatenate([pos_mtx.cpu().numpy(), np.ones([pos_mtx.shape[0], 1])], axis=1) # convert nx3 to nx4
                pos_mtx = pos_mtx @ gl_to_cv.T
                pos_mtx_half = np.concatenate([pos_mtx_half.cpu().numpy(), np.ones([pos_mtx_half.shape[0], 1])], axis=1) # convert nx3 to nx4
                pos_mtx_half = pos_mtx_half @ gl_to_cv.T
                nor_mtx = np.concatenate([nor_mtx.cpu().numpy(), np.zeros([pos_mtx.shape[0], 1])], axis=1) # convert nx3 to nx4
                nor_mtx = nor_mtx @ gl_to_cv.T
                for i in range(total_points):
                    x = int(pos_screen[i][0].item() + 0.5)
                    y = int(pos_screen[i][1].item() + 0.5)
                    # if y >= 0 and y < eye_mask.shape[1] and x >= 0 and x < eye_mask.shape[2] and eye_mask[0, y, x].item() < 0.5:
                    if y >= 0 and y < eye_mask.shape[1] and x >= 0 and x < eye_mask.shape[2] and eye_mask[0, y, x].item() < 0.5 and nor_mtx[i, 2] < 0:
                    # if y >= 0 and y < eye_mask.shape[1] and x >= 0 and x < eye_mask.shape[2] and eye_mask[0, y, x].item() < 0.5 and nor_mtx[i, 2] < 0 and nor_mtx[i, 1] < 0:
                        nomask_xyz.append(pos_mtx[i, :3])
                        nomask_xyz_half.append(pos_mtx_half[i, :3])
                        nomask_nor.append(nor_mtx[i, :3])
                    # if y >= 0 and y < eye_mask.shape[1] and x >= 0 and x < eye_mask.shape[2] and eye_mask[0, y, x].item() > 0.5 and nor_mtx[i, 2] < 0:
                    #     mask_xyz.append(pos_mtx[i, :3])
                    #     mask_xyz_half.append(pos_mtx_half[i, :3])
                    #     mask_nor.append(nor_mtx[i, :3])
                nomask_xyz = np.stack(nomask_xyz)
                nomask_xyz_arr.append(nomask_xyz)
                nomask_xyz_half = np.stack(nomask_xyz_half)
                nomask_xyz_half_arr.append(nomask_xyz_half)
                nomask_nor = np.stack(nomask_nor)
                nomask_nor_arr.append(nomask_nor)
                # mask_xyz = np.stack(mask_xyz)
                # mask_xyz_arr.append(mask_xyz)
                # mask_xyz_half = np.stack(mask_xyz_half)
                # mask_xyz_half_arr.append(mask_xyz_half)
                # mask_nor = np.stack(mask_nor)
                # mask_nor_arr.append(mask_nor)
                print('out_mask:', nomask_xyz.shape)
                # print('in_mask:', mask_xyz.shape)
                if step == 0:
                    self.p_eyeball.save_textured_mesh(eye_tex.cpu().numpy(), mtx.cpu().numpy(), self.eb_params.scale.cpu().numpy(), os.path.join(self.log_dir, '%04d.ply' % step))

        nomask_xyz_arr = np.array(nomask_xyz_arr, dtype=object)
        nomask_xyz_half_arr = np.array(nomask_xyz_half_arr, dtype=object)
        nomask_nor_arr = np.array(nomask_nor_arr, dtype=object)
        # mask_xyz_arr = np.array(mask_xyz_arr, dtype=object)
        # mask_xyz_half_arr = np.array(mask_xyz_half_arr, dtype=object)
        # mask_nor_arr = np.array(mask_nor_arr, dtype=object)
        np.savez(os.path.join(self.log_dir, 'sample_points.npz'), xyz=nomask_xyz_arr, nor=nomask_nor_arr, xyz_half=nomask_xyz_half_arr)
        # np.savez(os.path.join(self.log_dir, 'sample_points_im.npz'), xyz=mask_xyz_arr, nor=mask_nor_arr, xyz_half=mask_xyz_half_arr)

        if not is_init:
            return
        with torch.no_grad():
            default_env = utils.load_envimage_hdr(self.config['data']['env'])
            eye_tex = self.p_eyeball.get_mean_texture_torch()

            for step, (img, iris_mask, eye_mask, cam_pose, idx) in enumerate(tqdm(test_dataloader)):
                cam_pose = cam_pose.to(self.device)
                mtx = self.pose_assemble(self.eb_params.gaze[idx[0]], cam_pose[0], self.eb_params)
                # mtx = self.pose_assemble(self.get_gaze(idx[0]), cam_pose[0], self.eb_params)

                # proj
                pos_screen = self.renderer.project_points(mtx, iris_proxy_info.pos * iris_proxy_info.scale)
                proj_center = torch.mean(pos_screen[:, :2], dim=0) #/ iris_mask.shape[2]
                mask_xy = utils.mask_to_xy(iris_mask[0])
                gt_center = torch.mean(mask_xy, dim=0) #/ iris_mask.shape[2]

                seg = self.renderer.render_segment(glctx, mtx, iris_proxy_info)
                gt_seg = np.zeros([iris_mask.shape[1], iris_mask.shape[2], 3], dtype=np.float32)
                gt_seg[:, :, 1] = 1.0
                gt_seg = gt_seg * iris_mask[0].cpu().numpy()
                seg_c3 = seg.repeat(1, 1, 1, 3)
                seg_show = gt_seg * 0.5 + seg_c3[0].cpu().numpy() * 0.5

                color_blend = self.renderer.render_blend(glctx, mtx, eyeball_info, cornea_info, eye_tex, default_env, cornea_alpha=self.config['render']['cornea_alpha'])
                color_full = color_blend.cpu() * eye_mask + img * (1 - eye_mask)

                # draw proj center
                proj_center_np = proj_center.cpu().numpy()
                gt_center_np = gt_center.cpu().numpy()
                cv2.circle(seg_show, (int(proj_center_np[0] + 0.5), int(proj_center_np[1] + 0.5)), 4, (255, 0, 0), -1)  # red
                cv2.circle(seg_show, (int(gt_center_np[0] + 0.5), int(gt_center_np[1] + 0.5)), 4, (0, 0, 255), -1)      # blue

                color_merge = np.concatenate([seg_show, color_full[0].numpy(), img[0].numpy()], axis=1) # [H, W, C]
                utils.save_image(os.path.join(result_dir, 'merge_%03d.png' % self.dataset.select_list[idx[0]]), color_merge)
                

    def save_interploate(self, dataset, bg_path, inter_mode='ud'):
        result_dir = os.path.join(self.log_dir, 'interploate_' + inter_mode)
        mesh_dir = os.path.join(self.log_dir, 'mesh_' + inter_mode)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        # load eyelid images (hard code)
        bg_dir = os.path.join(bg_path, 'validations_inter_%s_geo' % inter_mode)
        bg_list = os.listdir(bg_dir)
        bg_num = len(bg_list) // 2
        bg_images = [utils.load_image(os.path.join(bg_dir, '00120000_%d.png' % i)) for i in range(bg_num)]
        bg_masks = [utils.load_image(os.path.join(bg_dir, '00120000_mask_%d.png' % i)) for i in range(bg_num)]

        if inter_mode == 'ud':
            dummy_gaze_np = utils.gen_gaze(pitch_max=np.pi/6, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
            # dummy_gaze_np = gen_gaze(pitch_max=0.8, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
        elif inter_mode == 'lr':
            dummy_gaze_np = utils.gen_gaze(pitch_max=0, pitch_num=1, yaw_max=np.pi/4, yaw_num=15).astype(np.float32)
            # dummy_gaze_np = gen_gaze(pitch_max=0, pitch_num=1, yaw_max=0.8, yaw_num=15).astype(np.float32)
        elif inter_mode == 'circle':
            dummy_gaze_np = utils.gen_circle_gaze(pitch_max=np.pi/6, yaw_max=np.pi/4, gaze_num=9).astype(np.float32)
            # dummy_gaze_np = gen_circle_gaze(pitch_max=0.8, yaw_max=0.8, gaze_num=9).astype(np.float32)
        else:
            print('[EXIT] Unsupported Inter Mode!')
            return
        dummy_gaze = torch.from_numpy(dummy_gaze_np).to(self.device)

        # save render results
        glctx = dr.RasterizeCudaContext()

        eyeball_info = self.p_eyeball.get_nvdiffrast_info()
        eyeball_info.to_tensor(self.device)
        cornea_info = self.p_eyeball.get_nvdiffrast_info_cornea()
        cornea_info.to_tensor(self.device)
        eyeball_info.set_scale(self.eb_params.scale)
        cornea_info.set_scale(self.eb_params.scale)

        with torch.no_grad():
            default_env = utils.load_envimage_hdr(self.config['data']['env'])
            eye_tex = self.p_eyeball.get_mean_texture_torch()

            data_num = len(dataset)
            gaze_num = dummy_gaze.shape[0]
            _, _, _, cam_pose, _ = dataset[data_num-1]
            # _, _, _, cam_pose, _ = dataset[0]
            for i in range(gaze_num):
                cam_pose = cam_pose.to(self.device)
                mtx = self.pose_assemble(dummy_gaze[i], cam_pose, self.eb_params)

                # save mesh of world space
                # fake_cam_pose = torch.tensor([
                #     [1, 0, 0, 0],
                #     [0, -1, 0, 0],
                #     [0, 0, -1, 0],
                #     [0, 0, 0, 1]
                # ], dtype=torch.float32).to(self.device)
                # world_mtx = self.pose_assemble(dummy_gaze[i], fake_cam_pose)
                self.p_eyeball.save_textured_mesh(eye_tex.cpu().numpy(), mtx.cpu().numpy(), self.eb_params.scale.cpu().numpy(), os.path.join(mesh_dir, '%04d.ply' % i))

                color_blend = self.renderer.render_blend(glctx, mtx, eyeball_info, cornea_info, eye_tex, default_env, cornea_alpha=self.config['render']['cornea_alpha'])
                color_save = color_blend[0].cpu().numpy()
                color_save = bg_images[i] * bg_masks[i] + color_save * (1 - bg_masks[i])

                utils.save_image(os.path.join(result_dir, '00120000_%d.png' % i), color_save)


    def save_interploate_both(self, dataset, bg_path, inter_mode='ud', use_split=True):
        result_dir = os.path.join(self.log_dir, 'interploate_' + inter_mode)
        mesh_dir = os.path.join(self.log_dir, 'mesh_' + inter_mode)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        # load eyelid images (hard code)
        bg_dir = os.path.join(bg_path, 'validations_inter_%s_geo' % inter_mode)
        bg_list = os.listdir(bg_dir)
        bg_num = len(bg_list) // 2
        bg_images = [utils.load_image(os.path.join(bg_dir, '00120000_%d.png' % i)) for i in range(bg_num)]
        bg_masks = [utils.load_image(os.path.join(bg_dir, '00120000_mask_%d.png' % i)) for i in range(bg_num)]

        gaze_arr = np.load(self.config['optim']['init_npz'])['gaze']
        # gaze_zero = np.mean(gaze_arr[:self.split], axis=0)
        gaze_zero = gaze_arr[0]

        if inter_mode == 'ud':
            dummy_gaze_np = utils.gen_gaze(pitch_max=np.pi/6, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
            # dummy_gaze_np = utils.gen_gaze(pitch_max=0.4, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
        elif inter_mode == 'lr':
            dummy_gaze_np = utils.gen_gaze(pitch_max=0, pitch_num=1, yaw_max=np.pi/4, yaw_num=15).astype(np.float32)
            # dummy_gaze_np = utils.gen_gaze(pitch_max=0, pitch_num=1, yaw_max=0.5, yaw_num=15).astype(np.float32)
        elif inter_mode == 'circle':
            dummy_gaze_np = utils.gen_circle_gaze(pitch_max=np.pi/6, yaw_max=np.pi/4, gaze_num=9).astype(np.float32)
            # dummy_gaze_np = utils.gen_circle_gaze(pitch_max=0.4, yaw_max=0.5, gaze_num=9).astype(np.float32)
        else:
            print('[EXIT] Unsupported Inter Mode!')
            return
        print(gaze_zero)
        dummy_gaze_np = dummy_gaze_np * 0.8 + gaze_zero     # assume gaze only control 80% eyelid movements
        dummy_gaze = torch.from_numpy(dummy_gaze_np).to(self.device)
        if use_split:
            zero_gaze = torch.zeros_like(dummy_gaze).to(dummy_gaze)
            both_move = torch.cat([dummy_gaze, dummy_gaze], dim=-1)
            os_move = torch.cat([dummy_gaze, zero_gaze], dim=-1)
            od_move = torch.cat([zero_gaze, dummy_gaze], dim=-1)
            dummy_gaze = torch.cat([both_move, os_move, od_move], dim=0)
        else:
            dummy_gaze = torch.cat([dummy_gaze, dummy_gaze], dim=-1)

        eb_params = EyeballParameter(self.dataset, self.device)
        eb_params.init_params(self.config['optim'], is_first=True)
        eb_params2 = EyeballParameter(self.dataset, self.device)
        eb_params2.init_params(self.config['optim'], is_first=False)


        # save render results
        glctx = dr.RasterizeCudaContext()

        eyeball_info = self.p_eyeball.get_nvdiffrast_info()
        eyeball_info.to_tensor(self.device)
        cornea_info = self.p_eyeball.get_nvdiffrast_info_cornea()
        cornea_info.to_tensor(self.device)

        with torch.no_grad():
            default_env = utils.load_envimage_hdr(self.config['data']['env'])
            eye_tex = self.p_eyeball.get_mean_texture_torch()

            data_num = len(dataset)
            gaze_num = dummy_gaze.shape[0]
            _, _, _, cam_pose, _ = dataset[data_num-1]
            for i in range(gaze_num):
                cam_pose = cam_pose.to(self.device)
                # OS
                mtx = self.pose_assemble(dummy_gaze[i, 0:2], cam_pose, eb_params)
                eyeball_info.set_scale(eb_params.scale)
                cornea_info.set_scale(eb_params.scale)
                self.p_eyeball.save_textured_mesh(eye_tex.cpu().numpy(), mtx.cpu().numpy(), eb_params.scale.cpu().numpy(), os.path.join(mesh_dir, 'os_%04d.ply' % i))
                os_color_blend, os_mask = self.renderer.render_blend(glctx, mtx, eyeball_info, cornea_info, eye_tex, default_env, cornea_alpha=self.config['render']['cornea_alpha'], ret_mask=True)

                # OD
                mtx = self.pose_assemble(dummy_gaze[i, 2:4], cam_pose, eb_params2)
                eyeball_info.set_scale(eb_params2.scale)
                cornea_info.set_scale(eb_params2.scale)
                self.p_eyeball.save_textured_mesh(eye_tex.cpu().numpy(), mtx.cpu().numpy(), eb_params2.scale.cpu().numpy(), os.path.join(mesh_dir, 'od_%04d.ply' % i))
                od_color_blend, od_mask = self.renderer.render_blend(glctx, mtx, eyeball_info, cornea_info, eye_tex, default_env, cornea_alpha=self.config['render']['cornea_alpha'], ret_mask=True)

                default_color = torch.ones(*od_color_blend.shape).to(od_color_blend)
                os_od_color_blend = os_color_blend * os_mask + od_color_blend * od_mask + default_color * (1 - os_mask) * (1 - od_mask)
                color_save = os_od_color_blend[0].cpu().numpy()
                color_save = bg_images[i] * bg_masks[i] + color_save * (1 - bg_masks[i])

                utils.save_image(os.path.join(result_dir, '00120000_%d.png' % i), color_save)



    def save_seg(self, idx):
        result_dir = os.path.join(self.log_dir, 'seg_results')
        os.makedirs(result_dir, exist_ok=True)
        
        glctx = dr.RasterizeCudaContext()
        iris_proxy_info = self.p_eyeball.get_nvdiffrast_info_iris_proxy()
        iris_proxy_info.to_tensor(self.device)
        iris_proxy_info.set_scale(self.scale)

        gaze_arr = np.load(self.config['optim']['init_npz'])['gaze']


        with torch.no_grad():
            _, _, _, cam_pose, _ = self.dataset[idx]
            gaze = torch.from_numpy(gaze_arr[idx])

            cam_pose = cam_pose.to(self.device)
            gaze = gaze.to(self.device)
            mtx = self.pose_assemble(gaze, cam_pose, self.eb_params)

            seg = self.renderer.render_segment(glctx, mtx, iris_proxy_info)
            seg = seg[0].cpu().numpy()
            print(seg.shape)
            utils.save_image(os.path.join(result_dir, 'seg_%03d.png' % idx), seg)
