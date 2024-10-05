import os
import logging
import argparse
import random
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from models.dataset import Dataset, MiniDataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, DeformNetwork, AppearanceNetwork, TopoNetwork, GazeDeformNetwork
from models.renderer import NeuSRenderer, DeformNeuSRenderer


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')
        self.gpu = torch.cuda.current_device()
        self.dtype = torch.get_default_dtype()

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # Backup codes and configs
        if mode[:5] == 'train':
            self.file_backup()
        if mode == 'train' or mode == 'image':
            self.dataset = Dataset(self.conf['dataset'])
        else:
            self.dataset = MiniDataset(self.conf['dataset'])
        self.iter_step = 0

        # set seed for exp
        self.random_seed = self.conf.get_int('train.random_seed', default=-1)
        if self.random_seed != -1:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            os.environ['PYTHONHASHSEED'] = str(self.random_seed)
            print('[Seed] set %d' % self.random_seed)

        self.use_normal = self.conf.get_bool('dataset.use_normal', default=False)
        self.use_gazeDA = self.conf.get_bool('dataset.use_gazeDA', default=False)
        self.use_exclude = self.conf.get_bool('dataset.use_exclude', default=False)
        self.use_disentangle = self.conf.get_bool('dataset.use_disentangle', default=False)
        print('[use_normal]:', self.use_normal)
        print('[use_gazeDA]:', self.use_gazeDA)
        print('[use_exclude]:', self.use_exclude)
        print('[use_disentangle]:', self.use_disentangle)

        # Gaze and Eyeball
        self.use_gaze = self.conf.get_bool('dataset.use_gaze', default=False)
        self.use_split = self.conf.get_bool('dataset.use_split', default=False)
        self.use_eyeball = self.conf.get_bool('dataset.use_eyeball', default=False)
        self.use_closing = self.conf.get_bool('train.use_closing', default=False)
        print('[use_gaze]:', self.use_gaze)
        print('[use_split]:', self.use_split)
        print('[use_eyeball]:', self.use_eyeball)
        print('[use_closing]:', self.use_closing)

        # Deform
        self.use_deform = self.conf.get_bool('train.use_deform')
        if self.use_deform:
            self.deform_dim = self.conf.get_int('model.deform_network.d_feature')
            if self.use_gaze:
                self.variance_dim = self.conf.get_int('model.gazedeform_network.d_variance', default=0)
                self.gazedeform_dim = self.conf.get_int('model.gazedeform_network.d_out') - self.variance_dim
                if self.variance_dim != 0:
                    print('[use_variance_codes]:', self.variance_dim)
                    self.variance_codes = torch.randn(self.dataset.n_images, self.variance_dim, requires_grad=True).to(self.device)
            else:
                self.deform_codes = torch.randn(self.dataset.n_images, self.deform_dim, requires_grad=True).to(self.device)
            
            if self.use_closing:
                self.closing_dim = self.gazedeform_dim
                self.closing_codes = torch.cat([torch.zeros(1, self.closing_dim).to(self.device), 
                                                torch.ones(1, self.closing_dim).to(self.device) * 0.5], dim=0)
            else:
                self.closing_dim = 0
            self.appearance_dim = self.conf.get_int('model.appearance_rendering_network.d_global_feature')
            self.appearance_codes = torch.randn(self.dataset.n_images, self.appearance_dim, requires_grad=True).to(self.device)

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.important_begin_iter = self.conf.get_int('model.neus_renderer.important_begin_iter')
        # Anneal
        self.max_pe_iter = self.conf.get_int('train.max_pe_iter')

        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.validate_idx = self.conf.get_int('train.validate_idx', default=-1)
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.test_batch_size = self.conf.get_int('test.test_batch_size')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight', default=0.1)
        self.mask_weight = self.conf.get_float('train.mask_weight', default=0.1)
        self.esp_weight = self.conf.get_float('train.esp_weight', default=0.1)
        self.disentangle_weight = self.conf.get_float('train.disentangle_weight', default=0.1)
        self.normal_penalty_weight = self.conf.get_float('train.normal_penalty_weight', default=3e-5)
        self.normal_penalty = self.conf.get_bool('train.normal_penalty', default=True)
        print(f"[INFO] Training setting -- normal penalty: {self.normal_penalty} -- weight: {self.normal_penalty_weight}")
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Deform
        if self.use_deform:
            if self.use_gaze:
                if self.use_split:
                    self.gazedeform_network_os = GazeDeformNetwork(**self.conf['model.gazedeform_network']).to(self.device)
                    self.gazedeform_network_od = GazeDeformNetwork(**self.conf['model.gazedeform_network']).to(self.device)
                else:
                    self.gazedeform_network = GazeDeformNetwork(**self.conf['model.gazedeform_network']).to(self.device)
            self.deform_network = DeformNetwork(**self.conf['model.deform_network']).to(self.device)
            self.topo_network = TopoNetwork(**self.conf['model.topo_network']).to(self.device)
        if self.use_gazeDA:
            if self.use_split:
                self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], eye_bbox=(self.dataset.os_bbox, self.dataset.od_bbox)).to(self.device)
            else:
                self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], eye_bbox=self.dataset.eye_bbox).to(self.device)
        else:
            self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # Deform
        if self.use_deform:
            self.color_network = AppearanceNetwork(**self.conf['model.appearance_rendering_network']).to(self.device)
        else:
            self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        # Deform
        if self.use_deform:
            self.renderer = DeformNeuSRenderer(self.report_freq,
                                     self.deform_network,
                                     self.topo_network,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     gazedeform_dim=self.gazedeform_dim + self.closing_dim,
                                     **self.conf['model.neus_renderer'])
        else:
            self.renderer = NeuSRenderer(self.sdf_network,
                                        self.deviation_network,
                                        self.color_network,
                                        **self.conf['model.neus_renderer'])

        # Load Optimizer
        params_to_train = []
        if self.use_deform:
            params_to_train += [{'name':'deform_network', 'params':self.deform_network.parameters(), 'lr':self.learning_rate}]
            params_to_train += [{'name':'topo_network', 'params':self.topo_network.parameters(), 'lr':self.learning_rate}]
            if self.use_gaze:
                if self.use_split:
                    params_to_train += [{'name':'gazedeform_network_os', 'params':self.gazedeform_network_os.parameters(), 'lr':self.learning_rate}]
                    params_to_train += [{'name':'gazedeform_network_od', 'params':self.gazedeform_network_od.parameters(), 'lr':self.learning_rate}]
                else:
                    params_to_train += [{'name':'gazedeform_network', 'params':self.gazedeform_network.parameters(), 'lr':self.learning_rate}]
                if self.variance_dim != 0:
                    params_to_train += [{'name':'variance_codes', 'params':self.variance_codes, 'lr':self.learning_rate}]
            else:
                params_to_train += [{'name':'deform_codes', 'params':self.deform_codes, 'lr':self.learning_rate}]
            if self.use_closing:
                params_to_train += [{'name':'closing_codes', 'params':self.closing_codes, 'lr':self.learning_rate}]
            params_to_train += [{'name':'appearance_codes', 'params':self.appearance_codes, 'lr':self.learning_rate}]
        params_to_train += [{'name':'sdf_network', 'params':self.sdf_network.parameters(), 'lr':self.learning_rate}]
        params_to_train += [{'name':'deviation_network', 'params':self.deviation_network.parameters(), 'lr':self.learning_rate}]
        params_to_train += [{'name':'color_network', 'params':self.color_network.parameters(), 'lr':self.learning_rate}]

        # Camera
        if self.dataset.camera_trainable:
            params_to_train += [{'name':'intrinsics_paras', 'params':self.dataset.intrinsics_paras, 'lr':self.learning_rate}]
            params_to_train += [{'name':'poses_paras', 'params':self.dataset.poses_paras, 'lr':self.learning_rate}]

        self.optimizer = torch.optim.Adam(params_to_train)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.mode == 'validate_pretrained':
                latest_model_name = 'pretrained.pth'
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def get_gaze(self, image_idx):
        if self.use_gaze:
            return self.dataset.get_gaze(image_idx)[None, ...]
        else:
            return None
        
    def get_gazedeform(self, gaze, alpha_ratio):
        if self.use_split:
            deform_code_os = self.gazedeform_network_os(gaze[:, 0:2], alpha_ratio)
            deform_code_od = self.gazedeform_network_od(gaze[:, 2:4], alpha_ratio)
            deform_code = torch.cat([deform_code_os, deform_code_od], dim=-1)
        else:
            deform_code = self.gazedeform_network(gaze, alpha_ratio)

        return deform_code

    def get_deformcode(self, image_idx, alpha_ratio):
        if self.use_gaze:
            deform_code = self.get_gazedeform(self.get_gaze(image_idx), alpha_ratio)
            if self.use_closing:
                closing_idx = 1 if image_idx in self.dataset.skip_frames else 0
                if self.use_split:
                    deform_code = torch.cat([deform_code[:, :self.gazedeform_dim], self.closing_codes[closing_idx][None, ...], 
                                            deform_code[:, self.gazedeform_dim:], self.closing_codes[closing_idx][None, ...]], dim=-1)
                else:
                    deform_code = torch.cat([deform_code, self.closing_codes[closing_idx][None, ...]], dim=-1)

            if self.variance_dim != 0:
                deform_code = torch.cat([deform_code, self.variance_codes[image_idx][None, ...]], dim=-1)
        else:
            deform_code = self.deform_codes[image_idx][None, ...]
        
        return deform_code
    
    def get_deformcode_interp(self, gaze, alpha_ratio, is_closed=False, variance_idx=0):
        assert self.use_gaze
        deform_code = self.get_gazedeform(gaze, alpha_ratio)
        if self.use_closing:
            closing_idx = 1 if is_closed else 0
            if self.use_split:
                deform_code = torch.cat([deform_code[:, :self.gazedeform_dim], self.closing_codes[closing_idx][None, ...], 
                                        deform_code[:, self.gazedeform_dim:], self.closing_codes[closing_idx][None, ...]], dim=-1)
            else:
                deform_code = torch.cat([deform_code, self.closing_codes[closing_idx][None, ...]], dim=-1)

        if self.variance_dim != 0:
            deform_code = torch.cat([deform_code, self.variance_codes[variance_idx][None, ...]], dim=-1)
        
        return deform_code

    def get_pseudocode(self, image_idx, deform_code):
        if self.use_disentangle:
            pseudo_code = torch.randn_like(deform_code).to(deform_code)

            # closing_code
            if self.use_closing:
                if self.use_split:
                    os_st = self.gazedeform_dim                                             # 16
                    os_ed = os_st + self.closing_dim                                        # 32
                    od_st = self.gazedeform_dim + self.closing_dim + self.gazedeform_dim    # 48
                    od_ed = od_st + self.closing_dim                                        # 64
                    assert os_st == 16 and os_ed == 32 and od_st == 48 and od_ed == 64
                    if self.iter_step >= 20000:
                        if image_idx in self.dataset.skip_frames:
                            # [closing frame] change closing_code, keep gazedeform_code
                            pseudo_code.data[:, 0:os_st] = deform_code.data[:, 0:os_st]
                            pseudo_code.data[:, os_ed:od_st] = deform_code.data[:, os_ed:od_st]
                        else:
                            if self.iter_step % 2 == 0:
                                # [open frame] change gazedeform_code, keep closing_code
                                pseudo_code.data[:, os_st:os_ed] = deform_code.data[:, os_st:os_ed]
                                pseudo_code.data[:, od_st:od_ed] = deform_code.data[:, od_st:od_ed]
                            #else
                                # [open frame] change both gazedeform_code and closing_code
                    else:
                        pseudo_code.data[:, os_st:os_ed] = self.closing_codes.data[0] + self.closing_codes.data[1] - deform_code.data[:, os_st:os_ed]
                        pseudo_code.data[:, od_st:od_ed] = self.closing_codes.data[0] + self.closing_codes.data[1] - deform_code.data[:, od_st:od_ed]
        else:
            pseudo_code = None

        assert pseudo_code.requires_grad == False, pseudo_code.requires_grad

        return pseudo_code

    def get_alpha_ratio(self):
        return max(min(self.iter_step/self.max_pe_iter, 1.), 0.)
    
    def get_decay_weight(self, w0, ratio, alpha):
        # ratio should be [0, 1]
        # decay from w0 to ratio * w0
        return w0 - (1 - ratio) * w0 * alpha

    def get_disentangle_weight(self, image_idx):
        if self.use_closing:
            if self.use_split:
                if self.iter_step < 20000:
                    scale_1 = 0.5
                    scale_23 = 10.0
                elif self.iter_step < self.max_pe_iter:
                    scale_1 = 10.0
                    scale_23 = 100.0 if image_idx not in self.dataset.skip_frames and self.iter_step % 2 == 0 else 10.0
                else:
                    scale_1 = 0.5
                    scale_23 = 10.0
            else:
                scale_1 = 0.5
                scale_23 = 10.0
        else:
            scale_1 = 0.5
            scale_23 = 1.0

        return scale_1, scale_23

    def get_more_sample(self):
        if self.use_closing:
            return self.iter_step < self.max_pe_iter
        return False

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        print(len(image_perm))

        for iter_i in tqdm(range(res_step)):
            # Deform
            if self.use_deform:
                image_idx = image_perm[self.iter_step % len(image_perm)]

                # Deform
                appearance_code = self.appearance_codes[image_idx][None, ...]
                # Anneal
                alpha_ratio = self.get_alpha_ratio()

                # Change closing_codes.requires_grad
                if self.use_closing and self.closing_codes.requires_grad == False:
                    if self.use_split:
                        if self.iter_step >= 20000:
                            self.closing_codes.requires_grad_()
                    else:
                        self.closing_codes.requires_grad_()
                
                deform_code = self.get_deformcode(image_idx, alpha_ratio)

                pseudo_code = self.get_pseudocode(image_idx, deform_code)
                
                if iter_i == 0:
                    print('The files will be saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    self.validate_observation_mesh(self.validate_idx)
                
                data = self.dataset.gen_random_rays_at(image_idx, self.batch_size, sample_more_eye=self.get_more_sample())
                if self.use_normal:
                    rays_o, rays_d, true_rgb, mask, eyemask, true_normal = data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10], data[:, 10:11], data[:, 11:14]
                else:
                    rays_o, rays_d, true_rgb, mask, eyemask = data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10], data[:, 10:11]


                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)
                mask_sum = mask.sum() + 1e-5
                
                
                if self.use_split:
                    input_eye_bbox = (self.dataset.os_bbox, self.dataset.od_bbox)
                else:
                    input_eye_bbox = self.dataset.eye_bbox

                if self.use_eyeball:
                    xyz_eyeball, nor_eyeball = self.dataset.get_esp_xyz_nor(image_idx)
                    if self.use_exclude:
                        render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                    alpha_ratio=alpha_ratio, iter_step=self.iter_step,
                                                    extra_samples=xyz_eyeball,
                                                    eye_bbox=input_eye_bbox,
                                                    deform_code2=pseudo_code,
                                                    shoot_eye=eyemask, inside_info=self.dataset.inside_info, gaze=self.get_gaze(image_idx))
                    else:
                        render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                    alpha_ratio=alpha_ratio, iter_step=self.iter_step,
                                                    extra_samples=xyz_eyeball, 
                                                    eye_bbox=input_eye_bbox,
                                                    deform_code2=pseudo_code,
                                                    gaze=self.get_gaze(image_idx))
                else:
                    render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=alpha_ratio, iter_step=self.iter_step,
                                                eye_bbox=input_eye_bbox,
                                                deform_code2=pseudo_code,
                                                gaze=self.get_gaze(image_idx))
                

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_o_error = render_out['gradient_o_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                # use normal penalty 
                pred_normal = render_out['pred_normal']
                grad_normal = render_out['grad_normal']
                weights = render_out['weights'].reshape(-1, 1).detach()

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_o_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-5, 1.0 - 1e-5), mask)
                if self.use_disentangle:
                    self.mask_weight = self.conf.get_float('train.mask_weight', default=0.1) + 0.1 * alpha_ratio

                if self.iter_step < self.max_pe_iter:
                    regular_scale = 10.0
                else:
                    regular_scale = 1.0

                loss = color_fine_loss +\
                    (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight) * regular_scale


                # normal_penalty
                if self.normal_penalty:
                    normal_error = (pred_normal - grad_normal) * weights 
                    normal_penalty = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum')
                    loss += normal_penalty * self.normal_penalty_weight
                
                if self.use_eyeball:
                    # eyeball sample points loss (sdf == 0)
                    sdf_eyeball = render_out['extra_sdf']
                    grad_eyeball = render_out['extra_grad']
                    grad_nor_eyeball = grad_eyeball / torch.linalg.norm(grad_eyeball, dim=-1, keepdim=True)
                    assert nor_eyeball.shape == grad_nor_eyeball.shape, (nor_eyeball.shape, grad_nor_eyeball.shape)
                    esp_xyz_loss = F.l1_loss(sdf_eyeball, torch.zeros_like(sdf_eyeball), reduction='sum') / sdf_eyeball.shape[0]
                    nor_dot = torch.sum(grad_nor_eyeball * nor_eyeball, dim=-1)
                    esp_nor_loss = F.l1_loss(nor_dot, torch.ones_like(nor_dot) * -1, reduction='sum') / grad_nor_eyeball.shape[0]
                    loss += esp_xyz_loss * self.esp_weight
                    loss += esp_nor_loss * self.esp_weight
                
                if self.use_disentangle:
                    coord_err1 = render_out['coord_err1']
                    coord_err2 = render_out['coord_err2']
                    coord_loss1 = F.l1_loss(coord_err1, torch.zeros_like(coord_err1), reduction='sum') #/ coord_err1.shape[0]
                    coord_loss2 = F.l1_loss(coord_err2, torch.zeros_like(coord_err2), reduction='sum') #/ coord_err2.shape[0]
                    
                    scale_1, scale_23 = self.get_disentangle_weight(image_idx)
                    weight_1 = self.disentangle_weight * scale_1
                    weight_23 = self.disentangle_weight * scale_23
                    if self.use_split:
                        coord_err3 = render_out['coord_err3']
                        coord_loss3 = F.l1_loss(coord_err3, torch.zeros_like(coord_err3), reduction='sum') #/ coord_err3.shape[0]
                        # balance weight_2 and weight_3
                        scale_2 = 1.5 if coord_loss2 > coord_loss3 else 0.5
                        scale_3 = 2.0 - scale_2
                        loss += (coord_loss1 * weight_1 + coord_loss2 * weight_23 * scale_2 + coord_loss3 * weight_23 * scale_3)
                    else:
                        loss += (coord_loss1 * weight_1 + coord_loss2 * weight_23)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                del color_fine_loss
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                del eikonal_loss
                del mask_loss
                if self.use_disentangle:
                    self.writer.add_scalar('Loss/coord_loss1', coord_loss1, self.iter_step)
                    self.writer.add_scalar('Loss/coord_loss2', coord_loss2, self.iter_step)
                    del coord_loss1
                    del coord_loss2
                    if self.use_split:
                        self.writer.add_scalar('Loss/coord_loss3', coord_loss3, self.iter_step)
                        del coord_loss3

                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The files have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss={} idx={} alpha_ratio={} lr={}'.format(self.iter_step, loss, image_idx,
                            alpha_ratio, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image(self.validate_idx)

                # if self.iter_step % self.val_mesh_freq == 0:
                #     self.validate_observation_mesh(self.validate_idx, resolution=512)

                self.update_learning_rate()
                
                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

            else:
                if self.iter_step == 0:
                    self.validate_mesh()
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

                rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                # use normal penalty 
                pred_normal = render_out['pred_normal']
                grad_normal = render_out['grad_normal']
                weights = render_out['weights'].reshape(-1, 1).detach()

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

                loss = color_fine_loss +\
                    eikonal_loss * self.igr_weight +\
                    mask_loss * self.mask_weight
                
                # normal_penalty
                if self.normal_penalty:
                    normal_error = (pred_normal - grad_normal) * weights
                    normal_penalty = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum')
                    loss += normal_penalty * self.normal_penalty_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                del color_fine_loss
                del eikonal_loss
                if self.mask_weight > 0.0:
                    self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                    del mask_loss
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The file have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_mesh()

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

        
    def get_image_perm(self):
        if self.use_closing and self.use_split and self.use_disentangle:
            if self.iter_step > self.warm_up_end and self.iter_step < self.max_pe_iter:
                base_list = np.arange(self.dataset.n_images).tolist()
                n_repeat = max(len(base_list) // len(self.dataset.skip_frames), 1) - 1
                merge_list = base_list + n_repeat * self.dataset.skip_frames  # make the raito of non-closed and closed samples approximate to 1:1
                np.random.shuffle(merge_list)
                return merge_list
            else:
                return torch.randperm(self.dataset.n_images)
        else:
            return torch.randperm(self.dataset.n_images)


    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])


    def update_learning_rate(self, scale_factor=1):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        learning_factor *= scale_factor

        current_learning_rate = self.learning_rate * learning_factor
        for g in self.optimizer.param_groups:
            if g['name'] in ['intrinsics_paras', 'poses_paras', 'depth_intrinsics_paras']:
                g['lr'] = current_learning_rate * 1e-1
            elif self.iter_step >= self.max_pe_iter and g['name'] == 'deviation_network':
                g['lr'] = current_learning_rate * 1.5
            elif self.use_closing and self.iter_step < 20000 and g['name'] in ['gazedeform_network_os', 'gazedeform_network_od']:
                g['lr'] = 0.0
            else:
                g['lr'] = current_learning_rate


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        if os.path.exists(os.path.join(self.base_exp_dir, 'recording')):
            print('recording exists! please delete first!')
            exit(0)
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        logging.info('File Saved')


    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # Deform
        if self.use_deform:
            if self.use_gaze:
                if self.use_split:
                    self.gazedeform_network_os.load_state_dict(checkpoint['gazedeform_network_os'])
                    self.gazedeform_network_od.load_state_dict(checkpoint['gazedeform_network_od'])
                else:
                    self.gazedeform_network.load_state_dict(checkpoint['gazedeform_network'])
                if self.variance_dim != 0:
                    self.variance_codes.data = torch.from_numpy(checkpoint['variance_codes']).to(self.device).data
            else:
                self.deform_codes.data = torch.from_numpy(checkpoint['deform_codes']).to(self.device).data
            if self.use_closing:
                self.closing_codes.data = torch.from_numpy(checkpoint['closing_codes']).to(self.device).data
            self.appearance_codes.data = torch.from_numpy(checkpoint['appearance_codes']).to(self.device).data
            self.deform_network.load_state_dict(checkpoint['deform_network'])
            self.topo_network.load_state_dict(checkpoint['topo_network'])
            logging.info('Use_deform True')
        self.dataset.intrinsics_paras.data = torch.from_numpy(checkpoint['intrinsics_paras']).to(self.device).data
        self.dataset.poses_paras.data = torch.from_numpy(checkpoint['poses_paras']).to(self.device).data
        # Camera
        if self.dataset.camera_trainable:
            self.dataset.intrinsics_paras.requires_grad_()
            self.dataset.poses_paras.requires_grad_()
        else:
            self.dataset.static_paras_to_mat()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')


    def save_checkpoint(self):
        # Deform
        if self.use_deform:
            if self.use_gaze:
                if self.use_split:
                    checkpoint = {
                        'gazedeform_network_os': self.gazedeform_network_os.state_dict(),
                        'gazedeform_network_od': self.gazedeform_network_od.state_dict(),
                        'deform_network': self.deform_network.state_dict(),
                        'topo_network': self.topo_network.state_dict(),
                        'sdf_network_fine': self.sdf_network.state_dict(),
                        'variance_network_fine': self.deviation_network.state_dict(),
                        'color_network_fine': self.color_network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'iter_step': self.iter_step,
                        'closing_codes': self.closing_codes.data.cpu().numpy() if self.use_closing else None,
                        'variance_codes': self.variance_codes.data.cpu().numpy() if self.variance_dim != 0 else None,
                        'appearance_codes': self.appearance_codes.data.cpu().numpy(),
                        'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                        'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                    }
                else:
                    checkpoint = {
                        'gazedeform_network': self.gazedeform_network.state_dict(),
                        'deform_network': self.deform_network.state_dict(),
                        'topo_network': self.topo_network.state_dict(),
                        'sdf_network_fine': self.sdf_network.state_dict(),
                        'variance_network_fine': self.deviation_network.state_dict(),
                        'color_network_fine': self.color_network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'iter_step': self.iter_step,
                        'closing_codes': self.closing_codes.data.cpu().numpy() if self.use_closing else None,
                        'variance_codes': self.variance_codes.data.cpu().numpy() if self.variance_dim != 0 else None,
                        'appearance_codes': self.appearance_codes.data.cpu().numpy(),
                        'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                        'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                    }
            else:
                checkpoint = {
                    'deform_network': self.deform_network.state_dict(),
                    'topo_network': self.topo_network.state_dict(),
                    'sdf_network_fine': self.sdf_network.state_dict(),
                    'variance_network_fine': self.deviation_network.state_dict(),
                    'color_network_fine': self.color_network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'iter_step': self.iter_step,
                    'deform_codes': self.deform_codes.data.cpu().numpy(),
                    'appearance_codes': self.appearance_codes.data.cpu().numpy(),
                    'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                    'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                }
        else:
            checkpoint = {
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
            }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>7d}.pth'.format(self.iter_step)))


    def validate_image(self, idx=-1, resolution_level=-1, mode='train', normal_filename='normals', rgb_filename='rgbs', depth_filename='depths'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        # Deform
        if self.use_deform:
            deform_code = self.get_deformcode(idx, self.get_alpha_ratio())
            appearance_code = self.appearance_codes[idx][None, ...]

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, eyemask = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        eyemask = eyemask.reshape(-1, 1).split(batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch, eyemask_batch in zip(rays_o, rays_d, eyemask):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            if self.use_deform:
                render_out = self.renderer.render(deform_code,
                                                appearance_code,
                                                rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=self.get_alpha_ratio(),
                                                iter_step=self.iter_step, gaze=self.get_gaze(idx))
                                                # shoot_eye=eyemask_batch, inside_info=self.dataset.inside_info)
                render_out['gradients'] = render_out['gradients_o']
            else:
                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                if self.iter_step >= self.important_begin_iter:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                else:
                    n_samples = self.renderer.n_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out['depth_map'] # Annotate it if you want to visualize estimated depth map!
            if feasible('depth_map'):
                out_depth_fine.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # Camera
            if self.dataset.camera_trainable:
                _, pose = self.dataset.dynamic_paras_to_mat(idx)
            else:
                pose = self.dataset.poses_all[idx]
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        depth_img = None
        if len(out_depth_fine) > 0:
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W, 1, -1])
            depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
            depth_img = np.uint8(depth_img)
        os.makedirs(os.path.join(self.base_exp_dir, rgb_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, normal_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, depth_filename), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        rgb_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        normal_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])
            
            if len(out_depth_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, depth_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                                        cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET))


    def validate_all_image(self, resolution_level=-1):
        idx_list = list(range(290, self.dataset.n_images))
        for image_idx in idx_list:
            self.validate_image(image_idx, resolution_level, 'test', 'validations_normals', 'validations_rgbs', 'validations_depths')
            print('Used GPU:', self.gpu)


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


    # Deform
    def validate_canonical_mesh(self, world_space=False, resolution=64, threshold=0.0, filename='meshes_canonical'):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_canonical_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=self.get_alpha_ratio())
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_canonical.ply'.format(self.iter_step)))

        logging.info('End')

    
    # Deform
    def validate_observation_mesh(self, idx=-1, world_space=False, resolution=64, threshold=0.0, filename='meshes', use_eye_bbox=False):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        
        # Deform
        deform_code = self.get_deformcode(idx, self.get_alpha_ratio())
        
        if not use_eye_bbox:
            bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
            bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        else:
            bound_min = torch.tensor(self.dataset.eye_bbox_min, dtype=self.dtype)
            bound_max = torch.tensor(self.dataset.eye_bbox_max, dtype=self.dtype)
            # bound_min = torch.tensor(self.dataset.os_bbox_min, dtype=self.dtype)
            # bound_max = torch.tensor(self.dataset.os_bbox_max, dtype=self.dtype)
            # bound_min = torch.tensor(self.dataset.od_bbox_min, dtype=self.dtype)
            # bound_max = torch.tensor(self.dataset.od_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=self.get_alpha_ratio(), gaze=self.get_gaze(idx))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_{}.ply'.format(self.iter_step, idx)))

        logging.info('End')


    # Deform
    def validate_all_mesh(self, world_space=False, resolution=64, threshold=0.0):
        idx_list = [0] + list(range(290, self.dataset.n_images))
        # idx_list = list(range(self.dataset.n_images))
        for image_idx in idx_list:
            self.validate_observation_mesh(image_idx, world_space, resolution, threshold, 'validations_meshes')
            print('Used GPU:', self.gpu)


    def bilinear_interpolate(self, code_0, code_1, offset):
        return code_0 * (1. - offset) + code_1 * offset

    def insert_closing(self, code_list, code_closing, idx, length=4):
        """
        Insert a closing frame into an animation sequence. 

        :param code_list: deform_code sequence
        :param code_closing: deform_code of closing frame
        :param idx: index of the closing frame (insertion position)
        :param length: frame number of closing animation
        """
        n_total = len(code_list)
        code_list[idx] = code_closing
        start_idx = idx - length
        end_idx = idx + length
        assert start_idx >= 0 and end_idx < n_total
        # start closing
        for i in range(start_idx+1, idx):
            offset = (i - start_idx) / length
            new_code = self.bilinear_interpolate(code_list[start_idx], code_closing, offset)
            code_list[i] = new_code
        # back to open
        for i in range(idx+1, end_idx):
            offset = (i - idx) / length
            new_code = self.bilinear_interpolate(code_closing, code_list[end_idx], offset)
            code_list[i] = new_code
        return code_list

    def validate_inter_mesh(self, world_space=False, resolution=64, threshold=0.0, filename='validations_inter', inter_mode='ud'):
        from utils import gen_gaze, gen_circle_gaze
        if inter_mode == 'ud':
            dummy_gaze_np = gen_gaze(pitch_max=np.pi/9, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
        elif inter_mode == 'lr':
            dummy_gaze_np = gen_gaze(pitch_max=0, pitch_num=1, yaw_max=np.pi/6, yaw_num=15).astype(np.float32)
        elif inter_mode == 'circle':
            dummy_gaze_np = gen_circle_gaze(pitch_max=np.pi/9, yaw_max=np.pi/6, gaze_num=9).astype(np.float32)
        else:
            print('[EXIT] Unsupported Inter Mode!')
            return
        dummy_gaze = torch.from_numpy(dummy_gaze_np).to(self.dtype).to(self.device)
        if self.use_split:
            zero_gaze = torch.zeros_like(dummy_gaze).to(dummy_gaze)
            both_move = torch.cat([dummy_gaze, dummy_gaze], dim=-1)
            os_move = torch.cat([dummy_gaze, zero_gaze], dim=-1)
            od_move = torch.cat([zero_gaze, dummy_gaze], dim=-1)
            dummy_gaze = torch.cat([both_move, os_move, od_move], dim=0)

            # # crossed eye
            # dummy_gaze = torch.cat([-dummy_gaze, dummy_gaze], dim=-1)

        print('before:', dummy_gaze[0], dummy_gaze[-1])
        dummy_gaze = dummy_gaze + self.get_gaze(0)
        print('after:', dummy_gaze[0], dummy_gaze[-1])

        with torch.no_grad():
            gaze_num = dummy_gaze.shape[0]
            print('Total Gaze:', gaze_num)
            code_list = []
            for idx in range(gaze_num):
                gaze = dummy_gaze[idx][None, ...]
                deform_code = self.get_deformcode_interp(gaze, self.get_alpha_ratio(), is_closed=False)
                code_list.append(deform_code)

            # interp_idx = 7
            interp_idx = None
            if self.use_closing and interp_idx is not None:
                # closed eye
                # if self.use_split:
                #     gaze_0 = torch.zeros([1, 4]).to(dummy_gaze).to(self.device)
                # else:
                #     gaze_0 = torch.zeros([1, 2]).to(dummy_gaze).to(self.device)
                gaze_0 = self.get_gaze(0)
                code_closing = self.get_deformcode_interp(gaze_0, self.get_alpha_ratio(), is_closed=True)
                code_list = self.insert_closing(code_list, code_closing, interp_idx, length=4)


            for idx, deform_code in enumerate(code_list):
                # if idx != 0 and idx != 14 and idx != 15 and idx != 29 and idx != 30 and idx != 44:
                #     continue
                gaze = dummy_gaze[idx][None, ...]

                bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
                bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
                
                vertices, triangles =\
                    self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                                alpha_ratio=self.get_alpha_ratio(), gaze=gaze)
                os.makedirs(os.path.join(self.base_exp_dir, filename + '_' + inter_mode), exist_ok=True)

                if world_space:
                    vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

                mesh = trimesh.Trimesh(vertices, triangles)
                mesh.export(os.path.join(self.base_exp_dir, filename + '_' + inter_mode, '{:0>8d}_{}.ply'.format(self.iter_step, idx)))

                logging.info('End')

    def validate_inter_mesh_code(self, world_space=False, resolution=64, threshold=0.0, filename='validations_inter_code', inter_idx=422, step=5):
        eye_dim = self.deform_dim - self.variance_dim
        print('eye_dim:', eye_dim)
        with torch.no_grad():
            src_deform_code = self.get_deformcode(0, self.get_alpha_ratio())
            tar_deform_code = self.get_deformcode(inter_idx, self.get_alpha_ratio())
            delta = (tar_deform_code - src_deform_code) / step
            code_list = []

            for i in range(step+1):
                temp = src_deform_code.clone()
                if self.use_split:
                    os_st = self.gazedeform_dim
                    os_ed = os_st + self.closing_dim
                    od_st = self.gazedeform_dim + self.closing_dim + self.gazedeform_dim
                    od_ed = od_st + self.closing_dim
                    # temp[:, os_st:os_ed] = temp[:, os_st:os_ed] + delta[:, os_st:os_ed] * i
                    temp[:, od_st:od_ed] = temp[:, od_st:od_ed] + delta[:, od_st:od_ed] * i
                else:
                    st = self.gazedeform_dim
                    ed = st + self.closing_dim
                    temp[:, st:ed] = temp[:, st:ed] + delta[:, st:ed] * i
                code_list.append(temp)

            for idx, deform_code in enumerate(code_list):
                # if idx != 0 and idx != 5:
                #     continue
                bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
                bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
                
                vertices, triangles =\
                    self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                                alpha_ratio=self.get_alpha_ratio(), gaze=self.get_gaze(inter_idx))
                os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

                if world_space:
                    vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

                mesh = trimesh.Trimesh(vertices, triangles)
                mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_{}.ply'.format(self.iter_step, idx)))

    # debug only
    def vis_topo_values(self, idx=-1, use_eye_bbox=False):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print(self.closing_codes)

        from utils import gen_gaze
        dummy_gaze_np = gen_gaze(pitch_max=np.pi/9, pitch_num=15, yaw_max=0, yaw_num=1).astype(np.float32)
        dummy_gaze = torch.from_numpy(dummy_gaze_np).to(self.dtype).to(self.device)
        if self.use_split:
            zero_gaze = torch.zeros_like(dummy_gaze).to(dummy_gaze)
            both_move = torch.cat([dummy_gaze, dummy_gaze], dim=-1)
            os_move = torch.cat([dummy_gaze, zero_gaze], dim=-1)
            od_move = torch.cat([zero_gaze, dummy_gaze], dim=-1)
            dummy_gaze = torch.cat([both_move, os_move, od_move], dim=0)

        print('before:', dummy_gaze[0], dummy_gaze[-1])
        dummy_gaze = dummy_gaze + self.get_gaze(0)
        print('after:', dummy_gaze[0], dummy_gaze[-1])

        with torch.no_grad():
            # Deform
            deform_code = self.get_deformcode(idx, self.get_alpha_ratio())

            change_code = self.get_gazedeform(dummy_gaze[0][None, ...], self.get_alpha_ratio())
            print(change_code.shape)

            if self.use_split:
                input_eye_bbox = (self.dataset.os_bbox, self.dataset.od_bbox)
            else:
                input_eye_bbox = self.dataset.eye_bbox

            self.renderer.vis_topo_values(deform_code, change_code, self.closing_codes, input_eye_bbox, self.get_alpha_ratio())



# This implementation is built upon NeuS: https://github.com/Totoro97/NeuS
if __name__ == '__main__':
    print('Welcome to NDR')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'image':
        runner.validate_all_image(resolution_level=1)
    elif args.mode[:8] == 'validate':
        if runner.use_deform:
            runner.validate_all_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
        else:
            runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'intergaze':
        runner.validate_inter_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold, inter_mode='ud')
        runner.validate_inter_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold, inter_mode='lr')
        runner.validate_inter_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold, inter_mode='circle')
        # runner.validate_inter_mesh_code(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'mesh':
        idx_list = [422, 472, 464]
        for idx in idx_list:
            runner.validate_observation_mesh(idx, world_space=False, resolution=512, threshold=args.mcube_threshold, filename='validations_meshes')
    elif args.mode == 'topo':
        runner.vis_topo_values(0)
        # runner.vis_topo_values(422)
