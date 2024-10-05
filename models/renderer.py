import torch
import torch.nn.functional as F
import numpy as np
import mcubes


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('Threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is built upon NeRF
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1) # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_eye_inside(pts, inside_info):
    x_min = inside_info['x_min']
    x_max = inside_info['x_max']
    eye_inside = (pts[:, 0] > x_min) & (pts[:, 0] < x_max)

    return eye_inside

def get_eye_bbox_mask(pts, eye_bbox):
    eye_bbox_torch = torch.from_numpy(eye_bbox).to(pts)
    mask = (pts > eye_bbox_torch[:3]).all(dim=-1, keepdim=True) & (pts < eye_bbox_torch[3:]).all(dim=-1, keepdim=True)
    
    return mask.float()

def comp_deform_codes(deform_code, change, gaze_dim=32):
    deform_code_reg1 = torch.cat([deform_code[:, :gaze_dim], change[:, gaze_dim:]], dim=-1) # same gaze, different others
    deform_code_reg2 = torch.cat([change[:, :gaze_dim], deform_code[:, gaze_dim:]], dim=-1) # different gaze, same others

    return deform_code_reg1, deform_code_reg2

def comp_deform_codes_split(deform_code, change, gaze_dim=16):
    # only one subcode changed
    deform_code_reg1 = torch.cat([deform_code[:, :2*gaze_dim], change[:, 2*gaze_dim:]], dim=-1) # same gaze, different others
    deform_code_reg2 = torch.cat([change[:, :gaze_dim], deform_code[:, gaze_dim:]], dim=-1) # different os, same others
    deform_code_reg3 = torch.cat([deform_code[:, :gaze_dim], change[:, gaze_dim:2*gaze_dim], deform_code[:, 2*gaze_dim:]], dim=-1) # different od, same others

    return deform_code_reg1, deform_code_reg2, deform_code_reg3

def comp_deform_codes_split2(deform_code, change, gaze_dim=16):
    # only one part unchanged
    deform_code_reg1 = torch.cat([change[:, :2*gaze_dim], deform_code[:, 2*gaze_dim:]], dim=-1) # same others, different gaze
    deform_code_reg2 = torch.cat([deform_code[:, :gaze_dim], change[:, gaze_dim:]], dim=-1) # same os, different others
    deform_code_reg3 = torch.cat([change[:, :gaze_dim], deform_code[:, gaze_dim:2*gaze_dim], change[:, 2*gaze_dim:]], dim=-1) # same od, different others

    return deform_code_reg1, deform_code_reg2, deform_code_reg3

# Deform
class DeformNeuSRenderer:
    def __init__(self,
                 report_freq,
                 deform_network,
                 ambient_network,
                 sdf_network,
                 deviation_network,
                 color_network,
                 begin_n_samples,
                 end_n_samples,
                 important_begin_iter,
                 n_importance,
                 up_sample_steps,
                 perturb,
                 gazedeform_dim):
        self.dtype = torch.get_default_dtype()
        # Deform
        self.deform_network = deform_network
        self.ambient_network = ambient_network
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.begin_n_samples = begin_n_samples
        self.end_n_samples = end_n_samples
        self.n_samples = self.begin_n_samples
        self.important_begin_iter = important_begin_iter
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.report_freq = report_freq
        self.gazedeform_dim = gazedeform_dim

        # points for disentanglement
        self.disent_points = self.get_disent_points('cuda')
        

    def get_disent_points(self, device):
        X = torch.linspace(-1, 1, 32)
        Y = torch.linspace(-1, 1, 32)
        Z = torch.linspace(-1, 1, 32)

        points = []
        for x in X:
            for y in Y:
                for z in Z:
                    points.append([x, y, z])
        
        points = torch.tensor(points).to(self.dtype).to(device)
        return points


    def update_samples_num(self, iter_step, alpha_ratio=0.):
        if iter_step >= self.important_begin_iter:
            self.n_samples = int(self.begin_n_samples*(1-alpha_ratio)+self.end_n_samples*alpha_ratio)


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples


    def cat_z_vals(self, deform_code, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False,
                alpha_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            # Deform
            pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
            ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
            new_sdf = self.sdf_network.sdf(pts_canonical, ambient_coord, alpha_ratio).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf


    def render_core(self,
                    deform_code,
                    appearance_code,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    deform_network,
                    ambient_network,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0,
                    alpha_ratio=0.,
                    extra_samples=None,
                    shoot_eye=None,
                    inside_info=None,
                    eye_bbox=None,
                    deform_code2=None,
                    gaze=None):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs_o = rays_d[:, None, :].expand(pts.shape) # view in observation space

        pts = pts.reshape(-1, 3)
        dirs_o = dirs_o.reshape(-1, 3)

        # Deform
        # observation space -> canonical space
        pts_canonical = deform_network(deform_code, pts, alpha_ratio)
        ambient_coord = ambient_network(deform_code, pts, alpha_ratio)

        if deform_code2 is not None:
            disent_pts = pts
            normal_coords = torch.cat([pts_canonical, ambient_coord], dim=-1)
            if isinstance(eye_bbox, tuple):
                in_os_bbox = get_eye_bbox_mask(disent_pts, eye_bbox[0])
                in_od_bbox = get_eye_bbox_mask(disent_pts, eye_bbox[1])
                mask_1 = 1-(1-in_os_bbox)*(1-in_od_bbox)
                mask_2 = 1-in_os_bbox
                mask_3 = 1-in_od_bbox

                deform_reg1, deform_reg2, deform_reg3 = comp_deform_codes_split(deform_code, deform_code2, self.gazedeform_dim)
                pts_c_reg1 = deform_network(deform_reg1, disent_pts, alpha_ratio)  # same os + od
                amb_c_reg1 = ambient_network(deform_reg1, disent_pts, alpha_ratio) # same os + od
                pts_c_reg2 = deform_network(deform_reg2, disent_pts, alpha_ratio)  # same (not os)
                amb_c_reg2 = ambient_network(deform_reg2, disent_pts, alpha_ratio) # same (not os)
                pts_c_reg3 = deform_network(deform_reg3, disent_pts, alpha_ratio)  # same (not od)
                amb_c_reg3 = ambient_network(deform_reg3, disent_pts, alpha_ratio) # same (not od)
                coord_err1 = (torch.cat([pts_c_reg1, amb_c_reg1], dim=-1) - normal_coords) * mask_1 / mask_1.sum()
                coord_err2 = (torch.cat([pts_c_reg2, amb_c_reg2], dim=-1) - normal_coords) * mask_2 / mask_2.sum()
                coord_err3 = (torch.cat([pts_c_reg3, amb_c_reg3], dim=-1) - normal_coords) * mask_3 / mask_3.sum()
                
                if deform_code.shape[-1] == 96 and alpha_ratio > 0.28571:
                    # only for use_closing + use_split when iter_step >= 20000
                    coord_err2 += (torch.cat([pts_c_reg2, amb_c_reg2], dim=-1) - normal_coords) * in_od_bbox / mask_2.sum() * 10.0
                    coord_err3 += (torch.cat([pts_c_reg3, amb_c_reg3], dim=-1) - normal_coords) * in_os_bbox / mask_3.sum() * 10.0
                
            else:
                in_eye_bbox = get_eye_bbox_mask(disent_pts, eye_bbox) # the point in eye_bbox
                mask_1 = in_eye_bbox
                mask_2 = 1-in_eye_bbox

                deform_reg1, deform_reg2 = comp_deform_codes(deform_code, deform_code2, self.gazedeform_dim)
                pts_c_reg1 = deform_network(deform_reg1, disent_pts, alpha_ratio)  # same eye
                amb_c_reg1 = ambient_network(deform_reg1, disent_pts, alpha_ratio) # same eye
                pts_c_reg2 = deform_network(deform_reg2, disent_pts, alpha_ratio)  # same (not eye)
                amb_c_reg2 = ambient_network(deform_reg2, disent_pts, alpha_ratio) # same (not eye)
                coord_err1 = (torch.cat([pts_c_reg1, amb_c_reg1], dim=-1) - normal_coords) * mask_1 / mask_1.sum()
                coord_err2 = (torch.cat([pts_c_reg2, amb_c_reg2], dim=-1) - normal_coords) * mask_2 / mask_2.sum()
                coord_err3 = None

        sdf_nn_output = sdf_network(pts_canonical, ambient_coord, alpha_ratio, gaze)
        sdf = sdf_nn_output[:, :1]

        if sdf_network.use_normal_pred:
            normal_vector = sdf_nn_output[:, 1:4]
            feature_vector = sdf_nn_output[:, 4:]
        else:
            feature_vector = sdf_nn_output[:, 1:]

        # Deform, gradients in observation space
        def gradient(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None, alpha_ratio=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio)
            amb_coord = ambient_network(deform_code, x, alpha_ratio)
            y = sdf_network.sdf(x_c, amb_coord, alpha_ratio, gaze)
            
            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o =  torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            # Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts =  torch.cat([grad_0, grad_1, grad_2], dim=1) # (batch_size, dim_out, dim_in)

            return gradient_o, gradient_pts, y

        # Deform
        # observation space -> canonical space
        if extra_samples is not None:
            n_extra_samples = extra_samples.shape[0]
            all_pts = torch.cat([extra_samples, pts], dim=0)
        else:
            all_pts = pts
        all_gradients_o, all_pts_jacobian, all_sdf = gradient(deform_network, ambient_network, sdf_network, deform_code, all_pts, alpha_ratio)
        if extra_samples is not None:
            extra_sample_sdf = all_sdf[:n_extra_samples, :]
            extra_sample_grad = all_gradients_o[:n_extra_samples, :]
            gradients_o = all_gradients_o[n_extra_samples:, :]
            pts_jacobian = all_pts_jacobian[n_extra_samples:, :]
        else:
            gradients_o = all_gradients_o
            pts_jacobian = all_pts_jacobian

        dirs_c = torch.bmm(pts_jacobian, dirs_o.unsqueeze(-1)).squeeze(-1) # view in observation space
        dirs_c = dirs_c / torch.linalg.norm(dirs_c, ord=2, dim=-1, keepdim=True)
        
        sampled_color = color_network(appearance_code, pts_canonical, gradients_o, \
            dirs_c, feature_vector, alpha_ratio).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs_o * gradients_o).sum(-1, keepdim=True) # observation

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()
        if shoot_eye is not None and inside_info is not None:
            eye_inside = get_eye_inside(pts, inside_info).reshape(batch_size, n_samples).to(self.dtype).detach()
            all_valid = torch.ones([batch_size, n_samples]).to(eye_inside).detach()
            merge_inside = shoot_eye * eye_inside + (1 - shoot_eye) * all_valid            # Merge Conditions
            alpha = alpha * merge_inside                                                 # Exclude invalid alpha
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # # weights_mask = weights.clone()
        # if shoot_eye is not None:
        #     eye_inside = get_eye_inside(pts, inside_info).reshape(batch_size, n_samples).to(self.dtype).detach()
        #     all_valid = torch.ones([batch_size, n_samples]).to(eye_inside).detach()
        #     merge_inside = shoot_eye * eye_inside + (1 - shoot_eye) * all_valid            # Merge Conditions
        #     weights = weights * merge_inside * alpha_ratio + weights * (1 - alpha_ratio)   # Exclude invalid alpha

        weights_sum = weights.sum(dim=-1, keepdim=True)

        # depth map
        depth_map = torch.sum(weights * mid_z_vals, -1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss, observation + canonical
        gradient_o_error = (torch.linalg.norm(gradients_o.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        relax_inside_sphere_sum = relax_inside_sphere.sum() + 1e-5
        gradient_o_error = (relax_inside_sphere * gradient_o_error).sum() / relax_inside_sphere_sum

        return {
            'pts': pts.reshape(batch_size, n_samples, 3),
            'pts_canonical': pts_canonical.reshape(batch_size, n_samples, 3),
            'relax_inside_sphere': relax_inside_sphere,
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients_o': gradients_o.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_sum': weights_sum,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_o_error': gradient_o_error,
            'inside_sphere': inside_sphere,
            'depth_map': depth_map,
            'pred_normal': normal_vector if sdf_network.use_normal_pred else None,
            'grad_normal': gradients_o / torch.linalg.norm(gradients_o, dim=-1, keepdim=True),
            'extra_sdf': extra_sample_sdf if extra_samples is not None else None,
            'extra_grad': extra_sample_grad if extra_samples is not None else None,
            'coord_err1': coord_err1 if deform_code2 is not None else None,
            'coord_err2': coord_err2 if deform_code2 is not None else None,
            'coord_err3': coord_err3 if deform_code2 is not None else None,
        }


    def render(self, deform_code, appearance_code, rays_o, rays_d, near, far, perturb_overwrite=-1,
            cos_anneal_ratio=0.0, alpha_ratio=0., iter_step=0, extra_samples=None, shoot_eye=None, inside_info=None, eye_bbox=None, deform_code2=None, gaze=None):
        self.update_samples_num(iter_step, alpha_ratio)
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if iter_step >= self.important_begin_iter and self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = pts.reshape(-1, 3)
                # Deform
                pts_canonical = self.deform_network(deform_code, pts, alpha_ratio)
                ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio)
                sdf = self.sdf_network.sdf(pts_canonical, ambient_coord, alpha_ratio, gaze).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(deform_code,
                                                  rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  alpha_ratio=alpha_ratio)

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(deform_code,
                                    appearance_code,
                                    rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.deform_network,
                                    self.ambient_network,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    alpha_ratio=alpha_ratio,
                                    extra_samples=extra_samples,
                                    shoot_eye=shoot_eye,
                                    inside_info=inside_info,
                                    eye_bbox=eye_bbox,
                                    deform_code2=deform_code2,
                                    gaze=gaze)

        weights = ret_fine['weights']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)


        return {
            'pts': ret_fine['pts'],
            'pts_canonical': ret_fine['pts_canonical'],
            'relax_inside_sphere': ret_fine['relax_inside_sphere'],
            'color_fine': ret_fine['color'],
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': ret_fine['weights_sum'],
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients_o': ret_fine['gradients_o'],
            'weights': weights,
            'gradient_o_error': ret_fine['gradient_o_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth_map': ret_fine['depth_map'],
            'pred_normal': ret_fine['pred_normal'],
            'grad_normal': ret_fine['grad_normal'],
            'extra_sdf': ret_fine['extra_sdf'],
            'extra_grad': ret_fine['extra_grad'],
            'coord_err1': ret_fine['coord_err1'],
            'coord_err2': ret_fine['coord_err2'],
            'coord_err3': ret_fine['coord_err3']
        }
    
    def extract_canonical_geometry(self, bound_min, bound_max, resolution, threshold=0.0, alpha_ratio=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts, alpha_ratio))

    
    def extract_observation_geometry(self, deform_code, bound_min, bound_max, resolution, threshold=0.0, alpha_ratio=0.0, gaze=None):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(self.deform_network(deform_code, pts,
                                                            alpha_ratio), self.ambient_network(deform_code, pts,
                                                            alpha_ratio), alpha_ratio, gaze))


    def vis_topo_values(self, deform_code, change_code, closing_codes, eye_bbox, alpha_ratio):
        pts = self.disent_points

        if isinstance(eye_bbox, tuple):
            in_os_bbox = get_eye_bbox_mask(pts, eye_bbox[0])
            in_od_bbox = get_eye_bbox_mask(pts, eye_bbox[1])

            print('os_num:', in_os_bbox.sum())
            print('od_num:', in_od_bbox.sum())
            print('others_num:', pts.shape[0] - in_os_bbox.sum() - in_od_bbox.sum())

            pts_c = self.deform_network(deform_code, pts, alpha_ratio)
            amb_c = self.ambient_network(deform_code, pts, alpha_ratio)

            # inverse deform_code
            random_code = torch.randn_like(deform_code).to(deform_code)
            # deform_code[:, 0:16] = change_code[:, 0:16] # random_code[:, 0:16]
            # deform_code[:, 16:32] = closing_codes[0] + closing_codes[1] - deform_code[:, 16:32]
            # deform_code[:, 32:48] = change_code[:, 16:32]
            deform_code[:, 48:64] = closing_codes[0] + closing_codes[1] - deform_code[:, 48:64]
            # deform_code[:, 64:] = random_code[:, 64:]

            pts2_c = self.deform_network(deform_code, pts, alpha_ratio)
            amb2_c = self.ambient_network(deform_code, pts, alpha_ratio)
            print((pts_c * in_os_bbox).sum(), (pts2_c * in_os_bbox).sum(), torch.abs((pts_c * in_os_bbox).sum() - (pts2_c * in_os_bbox).sum()))
            print((pts_c * in_od_bbox).sum(), (pts2_c * in_od_bbox).sum(), torch.abs((pts_c * in_od_bbox).sum() - (pts2_c * in_od_bbox).sum()))
            print((amb_c * in_os_bbox).sum(), (amb2_c * in_os_bbox).sum(), torch.abs((amb_c * in_os_bbox).sum() - (amb2_c * in_os_bbox).sum()))
            print((amb_c * in_od_bbox).sum(), (amb2_c * in_od_bbox).sum(), torch.abs((amb_c * in_od_bbox).sum() - (amb2_c * in_od_bbox).sum()))

        else:
            in_eye_bbox = get_eye_bbox_mask(pts, eye_bbox)
            pts_c = self.deform_network(deform_code, pts, alpha_ratio)
            amb_c = self.ambient_network(deform_code, pts, alpha_ratio)

            # inverse deform_code
            random_code = torch.randn_like(deform_code).to(deform_code)
            # deform_code[:, 0:32] = random_code[:, 32:64]
            deform_code[:, 32:64] = closing_codes[0] + closing_codes[1] - deform_code[:, 32:64]
            # deform_code[:, 64:] = random_code[:, 64:]

            pts2_c = self.deform_network(deform_code, pts, alpha_ratio)
            amb2_c = self.ambient_network(deform_code, pts, alpha_ratio)
            print((pts_c * in_eye_bbox).sum(), (pts2_c * in_eye_bbox).sum(), torch.abs((pts_c * in_eye_bbox).sum() - (pts2_c * in_eye_bbox).sum()))
            print((amb_c * in_eye_bbox).sum(), (amb2_c * in_eye_bbox).sum(), torch.abs((amb_c * in_eye_bbox).sum() - (amb2_c * in_eye_bbox).sum()))


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples


    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf


    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None] # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts)
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio) # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }


    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }


    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
