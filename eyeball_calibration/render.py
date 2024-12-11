import argparse
import os
import pathlib
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import rast_util
from nvdiffrec_render import renderutils as ru
import nvdiffrec_render.util

import nvdiffrast.torch as dr


# Refer to Nvdiffrec light.py
# https://github.com/NVlabs/nvdiffrec/blob/e7f2181b8a60eb8fedcdb4ad4d05bff3c0cf9bc1/render/light.py#L22
class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return nvdiffrec_render.util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = nvdiffrec_render.util.safe_normalize(nvdiffrec_render.util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

class CameraInfo():
    def __init__(self, fpath=None):
        if fpath is not None:
            with open(fpath, 'r') as f:
                cam_config = yaml.safe_load(f)
            self.h = cam_config['h']
            self.w = cam_config['w']
            self.fx = cam_config['fx']
            self.fy = cam_config['fy']
            self.cx = cam_config['cx']
            self.cy = cam_config['cy']
        else:
            self.h = None
            self.w = None
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None

    def get_proj_gl(self, near=0.1, far=1000.0, to_tensor=True):
        proj_np = np.array([
                [2*self.fx/self.w,                0,    1-2*self.cx/self.w,                       0],
                [               0, 2*self.fy/self.h,    2*self.cy/self.h-1,                       0],
                [               0,                0, (far+near)/(near-far), (2*far*near)/(near-far)],
                [               0,                0,                    -1,                       0]]).astype(np.float32)

        if to_tensor:
            return torch.from_numpy(proj_np)

        return proj_np

    def get_proj(self, to_tensor=True):
        proj_np = np.array([
                [-self.fx,       0, self.cx],   # negative for coordinates adaptation
                [      0, self.fy, self.cy],
                [      0,       0,       1]]).astype(np.float32)

        if to_tensor:
            return torch.from_numpy(proj_np)
        
        return proj_np

class SimpleRender():
    def __init__(self, camera_info, device):
        self.device = device
        self.camera_info = camera_info
        self.res_h = camera_info.h
        self.res_w = camera_info.w
        self.proj_gl = camera_info.get_proj_gl().to(device)
        self.proj = camera_info.get_proj().to(device)

        self.rast_out = None
        self.rast_out_db = None

    def reset_rast(self):
        self.rast_out = None
        self.rast_out_db = None

    def transform_vec(self, mtx, pos, v=1):
        t_mtx = torch.from_numpy(mtx).to(self.device) if isinstance(mtx, np.ndarray) else mtx
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(self.device) * v], dim=1)
        return torch.matmul(posw, t_mtx.t())

    def project_points(self, mtx, pos, ret_3d=False):
        pos_mtx = self.transform_vec(mtx, pos)[:, :3].contiguous()
        # reverse x-axis
        pos_screen = torch.matmul(pos_mtx, self.proj.t())
        pos_screen = pos_screen / pos_screen[:, 2:3]
        # reverse screen space
        # pos_screen[:, 0] = self.camera_info.w - pos_screen[:, 0]
        pos_screen[:, 1] = self.camera_info.h - pos_screen[:, 1]

        # # center
        # center = torch.mean(pos_screen[:, :2], dim=0) / self.camera_info.w

        # # radius
        # radius = torch.max(torch.max(pos_screen, dim=0)[0] - torch.min(pos_screen, dim=0)[0]) / self.camera_info.w
        # return center, radius

        if ret_3d:
            return pos_screen[:, :2], pos_mtx
        return pos_screen[:, :2]

    def render_blend(self, glctx, mtx, eyeball_info, cornea_info, tex, env, env_rot=None, cornea_alpha=0.2, vis_mask=None, ret_mask=False):
        self.reset_rast()   # reset rast_out and rast_out_db
        eyeball_color, is_eb = self.render_eyeball(glctx, mtx, eyeball_info, tex, env, env_rot, vis_mask)
        cornea_color, is_bg = self.render_cornea(glctx, mtx, cornea_info, env, env_rot, vis_mask)
        cornea_color = torch.where(is_bg, eyeball_color, cornea_color)
        blend_mask = 1 - (1 - is_eb) * is_bg
        # blend_mask = is_eb | (torch.logical_not(is_bg))
        # blend_mask = is_eb | (~is_bg)

        if ret_mask:
            return cornea_alpha * cornea_color + (1 - cornea_alpha) * eyeball_color, blend_mask

        return cornea_alpha * cornea_color + (1 - cornea_alpha) * eyeball_color

    def render_eyeball(self, glctx, mtx, eyeball_info, tex, env, env_rot=None, vis_mask=None):
        self.reset_rast()   # reset rast_out and rast_out_db

        pos = eyeball_info.pos * eyeball_info.scale
        nor = eyeball_info.nor
        uv = eyeball_info.uv
        pos_idx = eyeball_info.pos_idx

        # Transform
        pos_mtx = self.transform_vec(mtx, pos)[:, :3].contiguous()      # [N, 3]
        nor_mtx = self.transform_vec(mtx, nor, 0)[:, :3].contiguous()   # [N, 3]
        pos_clip = self.transform_vec(self.proj_gl, pos_mtx)[None, ...]    # [N, 4]

        # Rasterize
        r_tex, is_eb = self.render_texture(glctx, pos_clip, pos_idx, uv, pos_idx, tex)
        if env_rot is not None:
            nor_mtx = self.transform_vec(env_rot, nor_mtx, 0)[:, :3].contiguous()
        nor_mtx = nor_mtx[None, ...]
        r_nor, r_nord, _ = self.render_direction(glctx, pos_clip, pos_idx, nor_mtx)

        # Get diffuse env (cubemaps)
        specular_cubemaps = [env]
        while specular_cubemaps[-1].shape[1] > 16:
            specular_cubemaps += [cubemap_mip.apply(specular_cubemaps[-1])]
        env_diffuse = ru.diffuse_cubemap(specular_cubemaps[-1])
        r_env = dr.texture(env_diffuse[None, ...], r_nor, uv_da=r_nord, filter_mode='linear-mipmap-linear', boundary_mode='cube')
        
        # Shading
        # r_color = r_tex * r_env
        r_color = r_tex
        if vis_mask is not None:
            r_color = torch.where(vis_mask[..., -1:] == 1, r_color, r_tex)

        return r_color, is_eb

    def render_cornea(self, glctx, mtx, cornea_info, env, env_rot=None, vis_mask=None):
        self.reset_rast()   # reset rast_out and rast_out_db

        pos = cornea_info.pos * cornea_info.scale
        nor = cornea_info.nor
        uv = cornea_info.uv
        pos_idx = cornea_info.pos_idx

        # Transform
        pos_mtx = self.transform_vec(mtx, pos)[:, :3].contiguous()      # [N, 3]
        nor_mtx = self.transform_vec(mtx, nor, 0)[:, :3].contiguous()   # [N, 3]
        pos_clip = self.transform_vec(self.proj_gl, pos_mtx)[None, ...]    # [N, 4]

        # Reflect direction
        viewvec = pos_mtx # - camera_pos # assume camera position [0, 0, 0]
        reflvec = viewvec - 2.0 * nor_mtx * torch.sum(nor_mtx * viewvec, -1, keepdim=True) # Reflection vectors at vertices.
        reflvec = reflvec / torch.sum(reflvec**2, -1, keepdim=True)**0.5 # Normalize.
        if env_rot is not None:
            reflvec = self.transform_vec(env_rot, reflvec, 0)[:, :3].contiguous()
        reflvec = reflvec[None, ...]

        # Rasterize
        r_refl, r_refld, is_bg = self.render_direction(glctx, pos_clip, pos_idx, reflvec)

        # Shading
        r_color = dr.texture(env[np.newaxis, ...], r_refl, uv_da=r_refld, filter_mode='linear-mipmap-linear', boundary_mode='cube')

        return r_color, is_bg

    def render_segment(self, glctx, mtx, mesh_info, antialias=True):
        self.reset_rast()   # reset rast_out and rast_out_db
        pos = mesh_info.pos * mesh_info.scale
        pos_idx = mesh_info.pos_idx

        pos_mtx = self.transform_vec(mtx, pos)[:, :3].contiguous()      # [N, 3]
        pos_clip = self.transform_vec(self.proj_gl, pos_mtx)[None, ...]    # [N, 4]
        
        if self.rast_out is None:
            rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[self.res_h, self.res_w])
        else:
            rast_out, rast_out_db = self.rast_out, self.rast_out_db

        seg = torch.clamp(rast_out[..., -1:], 0, 1)
        if antialias:
            seg = dr.antialias(seg, rast_out, pos_clip, pos_idx)
        return seg

    def render_texture(self, glctx, pos_clip, pos_idx, uv, uv_idx, tex, use_mip=True, max_mip_level=9, antialias=True):
        # Rasterize.
        if self.rast_out is None:
            rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[self.res_h, self.res_w])
        else:
            rast_out, rast_out_db = self.rast_out, self.rast_out_db

        if use_mip:
            texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
            color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')

        # default_color = torch.ones(*color.shape).to(color)
        default_color = torch.zeros(*color.shape).to(color)
        mask = torch.clamp(rast_out[..., -1:], 0, 1)
        color = color * mask + default_color * (1 - mask) # Mask out background.
        if antialias:
            color = dr.antialias(color, rast_out, pos_clip, pos_idx)
        return color, mask

    def render_direction(self, glctx, pos_clip, pos_idx, direction, antialias=False):
        # Rasterize.
        if self.rast_out is None:
            rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[self.res_h, self.res_w])
        else:
            rast_out, rast_out_db = self.rast_out, self.rast_out_db

        direction, direction_d = dr.interpolate(direction, rast_out, pos_idx, rast_db=rast_out_db, diff_attrs='all') # Interpolated reflection vectors.
        direction = direction / (torch.sum(direction**2, -1, keepdim=True) + 1e-8)**0.5  # Normalize.
        if antialias:
            direction = dr.antialias(direction, rast_out, pos_clip, pos_idx)
        # Return
        return direction, direction_d, (rast_out[..., -1:] == 0)


if __name__ == '__main__':
    camera_info = CameraInfo('configs/camera.yaml')
    
    
