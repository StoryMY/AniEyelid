import os
import numpy as np
import torch
import trimesh
from PIL import Image
import utils

class MeshInfo():
    def __init__(self, pos_idx, vtx_pos, vtx_uv, vtx_nor):
        self.pos_idx = pos_idx
        self.pos = vtx_pos
        self.uv = vtx_uv
        self.nor = vtx_nor

    def to_tensor(self, device):
        self.pos_idx = utils.to_tensor(self.pos_idx, np.int32).to(device)
        self.pos = utils.to_tensor(self.pos).to(device)
        self.uv = utils.to_tensor(self.uv).to(device)
        self.nor = utils.to_tensor(self.nor).to(device)

    def set_scale(self, scale):
        self.scale = scale

class ParametricEyeball():
    def __init__(self, model_dir, device):
        # mesh
        self.eyeball = trimesh.load_mesh(os.path.join(model_dir, 'eyeball.obj'))
        self.cornea = trimesh.load_mesh(os.path.join(model_dir, 'cornea_z10_t24.obj'))
        self.pupil = trimesh.load_mesh(os.path.join(model_dir, 'pupil.obj'))
        self.iris_proxy = trimesh.load_mesh(os.path.join(model_dir, 'iris_proxy.obj'))

        # sample points
        self.sample_points, self.sample_normals = self.get_sample_points()

        # texture border
        self.texture = Image.open(os.path.join(model_dir, 'eye_texture_mini.png'))
        self.texture = np.array(self.texture) / 255   # normalize to [0, 1]
        _, self.crop_info = utils.center_crop(self.texture)
        self.texture_torch = torch.from_numpy(self.texture.astype(np.float32)).to(device)

    def get_sample_points(self):
        sample_points = []
        sample_normals = []
        # select eyeball vertices excluding iris
        for pt, vn in zip(self.eyeball.vertices, self.eyeball.vertex_normals):
            if (pt[0] ** 2 + pt[1] ** 2) > 0.25 and pt[2] < 1.18 and pt[2] > -0.4:
                sample_points.append(pt)
                sample_normals.append(vn)
        for pt, vn in zip(self.cornea.vertices, self.cornea.vertex_normals):
            sample_points.append(pt)
            sample_normals.append(vn)

        sample_points = np.stack(sample_points)
        sample_normals = np.stack(sample_normals)
        print('Sample points:', sample_points.shape)
        return sample_points, sample_normals

    def get_nvdiffrast_info(self):
        # merge eyeball and pupil mesh
        eyeball_pos_idx = self.eyeball.faces
        pupil_pos_idx = self.pupil.faces + self.eyeball.vertices.shape[0]
        pos_idx = np.concatenate([eyeball_pos_idx, pupil_pos_idx], axis=0)
        vtx_pos = np.concatenate([self.eyeball.vertices, self.pupil.vertices], axis=0)
        uv_idx = pos_idx
        vtx_uv = np.concatenate([self.eyeball.visual.uv, self.pupil.visual.uv], axis=0)
        vtx_nor = np.concatenate([self.eyeball.vertex_normals, self.pupil.vertex_normals], axis=0)

        return MeshInfo(pos_idx, vtx_pos, vtx_uv, vtx_nor)

    def get_nvdiffrast_info_cornea(self):
        pos_idx = self.cornea.faces
        vtx_pos = self.cornea.vertices
        uv_idx = pos_idx
        vtx_uv = self.cornea.visual.uv
        vtx_nor = self.cornea.vertex_normals

        return MeshInfo(pos_idx, vtx_pos, vtx_uv, vtx_nor)
    
    def get_nvdiffrast_info_iris_proxy(self):
        pos_idx = self.iris_proxy.faces
        vtx_pos = self.iris_proxy.vertices
        uv_idx = pos_idx
        vtx_uv = np.zeros([vtx_pos.shape[0], 2])
        vtx_nor = self.iris_proxy.vertex_normals

        return MeshInfo(pos_idx, vtx_pos, vtx_uv, vtx_nor)

    def save_textured_mesh(self, texture, mtx=None, scale=1.0, save_path='out.ply'):
        temp = (texture * 255)
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        self.eyeball.visual.material.image = Image.fromarray(temp)
        save_mesh = self.eyeball.copy()
        if mtx is not None:
            gl_to_cv = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            old_verts = np.concatenate([save_mesh.vertices * scale, np.ones([save_mesh.vertices.shape[0], 1])], axis=1) # convert nx3 to nx4
            new_verts = old_verts @ (gl_to_cv @ mtx).T
            save_mesh.vertices = new_verts[:, :3]
        # uvs = self.eyeball.visual.uv
        # colors = trimesh.visual.uv_to_interpolated_color(uvs, self.eyeball.visual.material.image)
        # self.eyeball.vertex_colors = colors
        save_mesh.export(save_path)

    def get_mean_texture(self):
        return self.texture
    
    def get_mean_texture_torch(self):
        return self.texture_torch
