import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

eps = 1e-6


# Anneal, Coarse-to-Fine Optimization part proposed by:
# Park, Keunhong, et al. Nerfies: Deformable neural radiance fields. CVPR 2021.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) * math.pi
        else:
            freq_bands = torch.linspace(2.**0.*math.pi, 2.**max_freq*math.pi, self.num_freqs)

        self.num_fns = len(self.kwargs['periodic_fns'])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs, alpha_ratio=0.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        start = 0
        if self.include_input:
            start = 1
        for i in range(self.num_freqs):
            output[:, (self.num_fns*i+start)*self.input_dims:(self.num_fns*(i+1)+start)*self.input_dims] *= (1.-math.cos(
                math.pi*(max(min(alpha_ratio*self.num_freqs-i, 1.), 0.))
            )) * .5
        return output


class DAGrid(nn.Module):
    def __init__(self, **kwargs):
        """
        """
        super(DAGrid, self).__init__()
        n_levels, base_resolution, desired_resolution = (
            kwargs['n_levels'],
            kwargs['base_resolution'], kwargs['desired_resolution']
        )
        b, n_features_per_level, bbox = (
            kwargs['b'], kwargs['n_features_per_level'], kwargs['bbox']
        )

        self.n_levels = n_levels
        if desired_resolution != -1:
            self.b = (desired_resolution / base_resolution) ** (1 / (n_levels - 1))
        else:
            self.b = b
        self.base_resolution = base_resolution # 16
        self.f = n_features_per_level # 2
        self.out_dim = self.f * self.n_levels # 32
        self.output_dim = self.out_dim + 3 # 35
        self.bounds = torch.from_numpy(np.array(bbox).reshape((2, 3))).float().cuda()
        
        self.os_bounds = None
        self.od_bounds = None
        self.eye_bounds = None
        if 'eye_bbox' in kwargs.keys() and kwargs['eye_bbox'] is not None:
            if isinstance(kwargs['eye_bbox'], tuple):
                self.os_bounds = torch.from_numpy(np.array(kwargs['eye_bbox'][0]).reshape((2, 3))).float().cuda()   # OS bbox
                self.od_bounds = torch.from_numpy(np.array(kwargs['eye_bbox'][1]).reshape((2, 3))).float().cuda()   # OD bbox
            else:
                self.eye_bounds = torch.from_numpy(np.array(kwargs['eye_bbox']).reshape((2, 3))).float().cuda()   # assume eye_bbox inside bbox

        self.size = (self.bounds[1] - self.bounds[0]).max().item()
        self.bounds[1] = self.bounds[1] - eps # [0, 1)
        self.offsets = [0]
        self.scales = []
        for i in range(self.n_levels):
            res = int((self.base_resolution) * (self.b**i))
            self.scales.append(res)
            n_entrys = int((res + 1) ** 3)
            self.offsets.append(self.offsets[-1] + n_entrys)

        anchors_ = self._init_anchors(freq_num=self.n_levels)
        self.data = torch.nn.Parameter(anchors_, requires_grad=True)
        if self.eye_bounds is not None or (self.os_bounds is not None and self.od_bounds is not None):
            gaze_diff = torch.zeros([2, anchors_.shape[0] * anchors_.shape[1]])
            self.diff = torch.nn.Parameter(gaze_diff, requires_grad=True)

        self.offsets_pos = torch.tensor([[0., 0., 0.],
                                     [0., 0., 1.],
                                     [0., 1., 0.],
                                     [0., 1., 1.],
                                     [1., 0., 0.],
                                     [1., 0., 1.],
                                     [1., 1., 0.],
                                     [1., 1., 1.]]).float().cuda()  # 8 x 3

        self.scales = torch.tensor(self.scales).cuda().float()
        self.offsets = torch.tensor(np.array(self.offsets)).cuda().long()

    def _init_anchors(self, freq_num=6):
        freq_bands = 2. ** torch.linspace(0., freq_num-1, freq_num)
        self.embed_fns = []
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
        anchors = []
        masks = []
        os_masks = []
        od_masks = []
        for i in range(self.n_levels):
            ti = [
                torch.linspace(self.bounds[0][_i], self.bounds[1][_i], self.scales[i] + 1)
                for _i in range(3)
            ]
            ti_ = torch.meshgrid(ti)
            xyz_ = torch.stack([ti_[0].flatten(), ti_[1].flatten(), ti_[2].flatten()], dim=-1) # N x 3
            anchors.append(xyz_)
            if self.eye_bounds is not None:
                mask_ = (xyz_ > self.eye_bounds[0]).all(dim=-1, keepdim=True) & (xyz_ < self.eye_bounds[1]).all(dim=-1, keepdim=True)   # inside eye_bbox
                masks.append(mask_)
            if self.os_bounds is not None:
                os_mask_ = (xyz_ > self.os_bounds[0]).all(dim=-1, keepdim=True) & (xyz_ < self.os_bounds[1]).all(dim=-1, keepdim=True)   # inside os_bbox
                os_masks.append(os_mask_)
            if self.od_bounds is not None:
                od_mask_ = (xyz_ > self.od_bounds[0]).all(dim=-1, keepdim=True) & (xyz_ < self.od_bounds[1]).all(dim=-1, keepdim=True)   # inside od_bbox
                od_masks.append(od_mask_)
        anchors = torch.cat(anchors, dim=0) # N' x 3
        if self.eye_bounds is not None:
            self.masks = torch.cat(masks, dim=0).float()
            print('Init GazeDA, mask:', self.masks.shape, '| valid:', torch.sum(self.masks))
        if self.os_bounds is not None:
            self.os_masks = torch.cat(os_masks, dim=0).float()
            print('Init GazeDA, os_mask:', self.os_masks.shape, '| valid:', torch.sum(self.os_masks))
        if self.od_bounds is not None:
            self.od_masks = torch.cat(od_masks, dim=0).float()
            print('Init GazeDA, od_mask:', self.od_masks.shape, '| valid:', torch.sum(self.od_masks))

        assert len(anchors) == self.offsets[-1], f'anchors dims not match offset dims, anchors: {len(anchors)}, offset[-1]: {self.offsets[-1]}.'
        return anchors

    def forward(self, xyz, alpha_ratio, gaze=None):
        xyz_ = torch.max(torch.min(xyz, self.bounds[1]), self.bounds[0])
        xyz_ = (xyz_ - self.bounds[None, 0]) / self.size
        xyz_ = xyz_[None].repeat(self.n_levels, 1, 1) # N x 3  -> n_level x N x 3
        float_xyz = xyz_ * self.scales[:, None, None]
        int_xyz = (float_xyz[:, :, None] + self.offsets_pos[None, None]).long() # n_level x N x 8 x 3
        offset_xyz = float_xyz - int_xyz[:, :, 0]   # n_level x N x 3

        ind = torch.zeros_like(int_xyz[..., 0])

        sh = self.n_levels
        ind[:sh] = int_xyz[:sh, ..., 0] * ((self.scales[:sh] + 1)**2)[:, None, None] + \
                int_xyz[:sh, ..., 1] * ((self.scales[:sh] + 1))[:, None, None] + \
                int_xyz[:sh, ..., 2]
        nl = self.n_levels
        
        ind = ind.reshape(nl, -1)
        ind += self.offsets[:-1, None]
        ind = ind.reshape(-1)

        val = torch.gather(self.data, 0, ind[:, None].repeat(1, 3))
        if gaze is not None and alpha_ratio > 0.99:
            if self.eye_bounds is not None:
                diff = (gaze @ self.diff).reshape((-1, 3))
                diff = torch.gather(diff, 0, ind[:, None].repeat(1, 3))
                mask = torch.gather(self.masks, 0, ind[:, None])
                val = val + diff * mask

            elif self.os_bounds is not None and self.od_bounds is not None:
                os_diff = (gaze[:, 0:2] @ self.diff).reshape((-1, 3))
                os_diff = torch.gather(os_diff, 0, ind[:, None].repeat(1, 3))
                os_mask = torch.gather(self.os_masks, 0, ind[:, None])

                od_diff = (gaze[:, 2:4] @ self.diff).reshape((-1, 3))
                od_diff = torch.gather(od_diff, 0, ind[:, None].repeat(1, 3))
                od_mask = torch.gather(self.od_masks, 0, ind[:, None])
                val = val + os_diff * os_mask + od_diff * od_mask
        val = val.reshape(nl, -1, 8, 3)     # n_level x N x 8 x 3

        # binary interploate (1, 0) + (-1, 1) * x
        weights_xyz = torch.clamp((1 - self.offsets_pos[None, None]) + (2 * self.offsets_pos[None, None] - 1.) * offset_xyz[:, :, None], min=0., max=1.)
        weights_xyz = weights_xyz[..., 0] * weights_xyz[..., 1] * weights_xyz[..., 2]

        val = torch.cat(
            [
                self.embed_fns[_i](val[_i//2,:,:])  # N x 3, (N x level x 3) x (2 * 8)
                for _i in range(len(self.embed_fns))
            ], 
            dim=-1
        ) # level x N x 8 x 6
        val = val.view(-1, 8, self.n_levels, self.f) # N x 8 x level x 6
        val = val.permute(0, 2, 1, 3) # N x level x 8 x 6

        # alpha_ratio
        start = 0
        speed_scale = 1.0 # 1.5
        alpha_ratio_scale = min(alpha_ratio * speed_scale, 1.0) 
        for i in range(self.n_levels):
            val[:, (i+start):((i+1)+start)] *= (1.-math.cos(
                math.pi*(max(min(alpha_ratio_scale*self.n_levels-i, 1.), 0.))
            )) * .5

        weights_xyz = weights_xyz.permute(1, 0, 2) # level x N x 8 --> N x level x 8
        val  = (weights_xyz[..., None] * val).sum(dim=-2)
        val = val.reshape(-1, self.out_dim)
        # 
        val = torch.cat([xyz, val], dim=-1)
        return val

def get_embedder(cfg, input_dims=3):
    if cfg.type == 'frequency':
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': cfg.freq-1,
            'num_freqs': cfg.freq,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        def embed(x, alpha_ratio, eo=embedder_obj): return eo.embed(x, alpha_ratio)
        return embed, embedder_obj.out_dim
    elif cfg.type == 'deformable_anchor_grid':
        embedder = DAGrid(
                n_levels=cfg.n_levels,
                base_resolution=cfg.base_resolution,
                n_features_per_level=cfg.n_features_per_level,
                desired_resolution=cfg.desired_resolution,
                b=cfg.b,
                bbox=cfg.bbox,
                eye_bbox=cfg.eye_bbox)
        return embedder, embedder.output_dim
    else:
        assert False, f'Unknown embedder type: {cfg.type}'
    