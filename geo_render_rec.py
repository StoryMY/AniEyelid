import openmesh as om
import os
import torch
import numpy as np
import cv2
from glob import glob
import sys

from models.LieAlgebra import se3
from renderer import RenderUtils

if len(sys.argv) != 5:
    print(
        "Give the path to the original data as the first argument,\n"
        "then the directory containing results, and the number of training iteration as the last.\n"
        "For example, execute this program by running:\n"
        "    python geo_render_ani.py <dataset_dir> <exp_dir> <iter_step> <start_index>\n")
    exit()
origin_data_path_ = sys.argv[1]
results_path_ = sys.argv[2]
iter_num_ = sys.argv[3]
st_num_ = sys.argv[4]

# scale factor of depth
# e.g. a pixel value of 1000 in the depth image corresponds to a distance of 1 meter from the camera.
depth_scale_ = 100.

# parameters of rendering
strength_ = 0.9
light_dire_ = np.array([0.0, 0.0, -1.0])
ambient_strength_ = 0.4 * strength_
light_strength_ = 0.6 * strength_



def project_mesh_vps(world_vps, camera_dict):
    ex_mat = camera_dict["ExterMat"]
    in_mat = camera_dict["InterMat"]
    cam_reso = camera_dict["CameraReso"]

    cam_w = cam_reso[0]
    cam_h = cam_reso[1]
    ex_Rmat = ex_mat[:3, :3]
    ex_Tvec = ex_mat[:3, 3:]
    
    fx = in_mat[0, 0]
    fy = in_mat[1, 1]
    cx = in_mat[0, 2]
    cy = in_mat[1, 2]
    
    cam_vps = ex_Rmat.dot(world_vps.T) + ex_Tvec
    pixel_x = fx * (cam_vps[0, :] / cam_vps[2, :]) + cx
    pixel_y = fy * (cam_vps[1, :] / cam_vps[2, :]) + cy
    
    vps_status = (pixel_x > 0) * (pixel_x < cam_w) * (pixel_y > 0) * (pixel_y < cam_h)
    proj_pixel = np.stack([pixel_x, pixel_y], axis=1)
    
    return proj_pixel, cam_vps[2, :], vps_status


def render_tex_mesh_func(fv_indices, tri_uvs, tri_normals, tex_img, vps, camera_dict):
    proj_pixels, z_vals, v_status = project_mesh_vps(vps, camera_dict)
    tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6) 
    tri_z_vals = z_vals[fv_indices] # [n_f, 3]
    tri_status = (v_status[fv_indices]).all(axis=1) # [n_f]

    cam_w = camera_dict["CameraReso"][0]
    cam_h = camera_dict["CameraReso"][1]
    ex_mat = camera_dict["ExterMat"]
    
    depth_img = np.ones((cam_h, cam_w), np.float32) * 100.0
    rgb_img = np.zeros((cam_h, cam_w, 3), np.float32)
    mask_img = np.zeros((cam_h, cam_w), np.int32)
    
    w_light_dx = light_dire_[0]
    w_light_dy = light_dire_[1]
    w_light_dz = light_dire_[2]
    
    c_light_dx = ex_mat[0, 0] * w_light_dx + ex_mat[0, 1] * w_light_dy + ex_mat[0, 2] * w_light_dz
    c_light_dy = ex_mat[1, 0] * w_light_dx + ex_mat[1, 1] * w_light_dy + ex_mat[1, 2] * w_light_dz
    c_light_dz = ex_mat[2, 0] * w_light_dx + ex_mat[2, 1] * w_light_dy + ex_mat[2, 2] * w_light_dz
    
    ambient_strength = ambient_strength_
    light_strength = light_strength_
    
    RenderUtils.render_tex_mesh(
        tri_normals, tri_uvs, tri_proj_pixels, tri_z_vals, tri_status, tex_img, depth_img, rgb_img, mask_img,
        c_light_dx,c_light_dy,c_light_dz,ambient_strength,light_strength
    )
    
    depth_img[mask_img < 0.5] = 0
    return rgb_img, depth_img, mask_img


def render_mesh(mesh_path, tex_img, scale_mat, extrinsic, camera_dict, save_dir, base_name):
    om_mesh = om.read_trimesh(mesh_path)

    # Vertex Position
    vps = om_mesh.points()
    vps = np.concatenate([vps, np.ones((vps.shape[0], 1))], 1)
    vps = np.matmul(scale_mat, np.matmul(extrinsic,vps.transpose(1,0)))[:3,:].transpose(1,0)

    n_f = om_mesh.n_faces()
    fv_indices = om_mesh.face_vertex_indices()
    fh_indices = om_mesh.face_halfedge_indices()
    
    # Face texture2D UV
    he_uv = om_mesh.halfedge_texcoords2D()
    tri_uvs = (he_uv[fh_indices]).reshape(n_f, 6)
    
    # Normal
    om_mesh.request_face_normals()
    om_mesh.request_vertex_normals()
    om_mesh.update_normals()
    vns = om_mesh.vertex_normals()
    tri_normals = (vns[fv_indices]).reshape(n_f, 9)
    
    rgb_img, depth_img, mask_img = render_tex_mesh_func(
        fv_indices, tri_uvs, tri_normals, tex_img, vps, camera_dict
    )
        
    cv2.imwrite(save_dir + "rgb_%s.png"%base_name, (rgb_img * 255)[:,:,::-1])
    cv2.imwrite(save_dir + "depth_%s.png"%base_name, (depth_img * depth_scale_).astype(np.uint16))
    cv2.imwrite(save_dir + "mask_%s.png"%base_name, (mask_img).astype(np.uint8))


def render_color_mesh_func(fv_indices, tri_colors, tri_normals, vps, camera_dict, extrinsic, bg_img=None):    
    proj_pixels, z_vals, v_status = project_mesh_vps(vps, camera_dict)
    tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6) # [n_f, 6]
    tri_z_vals = z_vals[fv_indices] # [n_f, 3]
    # tri_status = (v_status[fv_indices]).all(axis=1) # [n_f]
    tri_status = (v_status[fv_indices]).any(axis=1) # [n_f]

    cam_w = camera_dict["CameraReso"][0]
    cam_h = camera_dict["CameraReso"][1]
    ex_mat = camera_dict["ExterMat"]
    
    depth_img = np.ones((cam_h, cam_w), np.float32) * 200.0
    if bg_img is None:
        # rgb_img = np.zeros((cam_h, cam_w, 3), np.float32)
        rgb_img = np.ones((cam_h, cam_w, 3), np.float32)
    else:
        rgb_img = bg_img.copy()
    mask_img = np.zeros((cam_h, cam_w), np.int32)
    
    # light = -np.matmul(np.transpose(extrinsic[:3,:3]), extrinsic[:3,-1])
    # light = -np.matmul(extrinsic[:3,:3], extrinsic[:3,-1])
    light = np.array([0.0, -1.0, -1.732])   # "-" for render_func.cpp implementation
    # light = np.array([0.0, 0.0, -1.0])
    light /= np.linalg.norm(light)  # + 1e-5, avoid 0

    c_light_dx = light[0]
    c_light_dy = light[1]
    c_light_dz = light[2]
    
    ambient_strength = ambient_strength_
    light_strength = light_strength_
    
    RenderUtils.render_color_mesh(
        tri_normals, tri_colors, tri_proj_pixels, tri_z_vals, tri_status, depth_img, rgb_img, mask_img,
        c_light_dx,c_light_dy,c_light_dz,ambient_strength,light_strength
    )
    depth_img[mask_img < 0.5] = 0.0
    return rgb_img, depth_img, mask_img


def render_color_mesh(mesh_path, scale_mat, extrinsic, camera_dict, save_dir, detph_save_dir, base_name):
                        
    om_mesh = om.read_trimesh(mesh_path)
    
    # Vertex Position
    vps = om_mesh.points()
    vps = np.concatenate([vps, np.ones((vps.shape[0], 1))], 1)
    vps = np.matmul(scale_mat, np.matmul(extrinsic, vps.transpose(1,0)))[:3,:].transpose(1,0)
    # for i in range(vps.shape[0]):
    #     om_mesh.set_point(om_mesh.vertex_handle(i), vps[i]) # update points, otherwise the normals are wrong

    n_f = om_mesh.n_faces()
    fv_indices = om_mesh.face_vertex_indices()
    
    # Normal
    om_mesh.request_face_normals()
    om_mesh.request_vertex_normals()
    om_mesh.update_normals()
    vns = om_mesh.vertex_normals()
    vns = np.concatenate([vns, np.zeros((vns.shape[0], 1))], 1)
    vns = np.matmul(extrinsic, vns.transpose(1,0))[:3,:].transpose(1,0)
    tri_normals = (vns[fv_indices]).reshape(n_f, 9)
    
    # Color
    vcs = np.array([[238., 233., 233.]]).repeat(om_mesh.n_vertices(), 0) / 255. # Snow2, WhiteSmoke
    vcs = np.ascontiguousarray(vcs)
    tri_colors = (vcs[fv_indices]).reshape(n_f, 9) 
    
    rgb_img, depth_img, mask_img = render_color_mesh_func(
        fv_indices, tri_colors, tri_normals, vps, camera_dict, extrinsic
    )
    
    cv2.imwrite(save_dir + "%s.png"%base_name, (rgb_img * 255)[:,:,::-1])
    cv2.imwrite(detph_save_dir + "%s.png"%base_name, (depth_img * depth_scale_).astype(np.uint16))
    # cv2.imwrite(save_dir + "depth_%s.png"%base_name, (depth_img * depth_scale_).astype(np.uint16))
    # cv2.imwrite(save_dir + "mask_%s.png"%base_name, (mask_img).astype(np.uint8))

# This function is based upon IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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


# This implementation is built upon StereoPIFu: https://github.com/CrisHY1995/StereoPIFu_Code
if __name__ == '__main__':
    images_lis = sorted(glob(os.path.join(origin_data_path_, 'rgb/*.jpg')))
    if len(images_lis) == 0:
        images_lis = sorted(glob(os.path.join(origin_data_path_, 'rgb/*.png')))
    data_num = len(images_lis)
    if data_num == 0:
        print('No data! The format of images must be jpg or png!')
        exit()
    img = cv2.imread(images_lis[0])
    H_, W_ = img.shape[0], img.shape[1]
    origin_cameras_path = os.path.join(origin_data_path_, 'cameras_sphere.npz')
    meshes_path = results_path_ + 'validations_meshes/' + str(iter_num_).zfill(8) + '_'
    
    side_view_angle = 0
    if side_view_angle != 0:
        save_path = results_path_ + 'validations_geo_%d/' % side_view_angle + str(iter_num_).zfill(8) + '_'
        os.makedirs(results_path_ + 'validations_geo_%d/' % side_view_angle, exist_ok=True)
        depth_save_path = results_path_ + 'validations_depth_%d/' % side_view_angle + str(iter_num_).zfill(8) + '_'
        os.makedirs(results_path_ + 'validations_depth_%d/' % side_view_angle, exist_ok=True)
    else:
        save_path = results_path_ + 'validations_geo/' + str(iter_num_).zfill(8) + '_'
        os.makedirs(results_path_ + 'validations_geo/', exist_ok=True)
        depth_save_path = results_path_ + 'validations_depth/' + str(iter_num_).zfill(8) + '_'
        os.makedirs(results_path_ + 'validations_depth/', exist_ok=True)

    # load scale_mat
    origin_cameras = np.load(origin_cameras_path)
    scale_mats_np = [origin_cameras['scale_mat_%d' % idx].astype(np.float32) for idx in range(data_num)]
    scale_mats = np.stack(scale_mats_np)
    scale_mats[:, :3, 3] = 0
    world_mats_np = [origin_cameras['world_mat_%d' % idx].astype(np.float32) for idx in range(data_num)]
    world_mats = np.stack(world_mats_np)
    intrinsics_all = []
    poses_all = []
    
    # render side view
    side_view_rmat, _ = cv2.Rodrigues(np.array([0, side_view_angle / 180 * np.pi, 0]))
    side_view_4x4 = np.eye(4, 4)
    side_view_4x4[:3, :3] = side_view_rmat
    if side_view_angle != 0:
        side_view_4x4[0, 3] = 0.1   # x translate

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        poses_all.append(np.linalg.inv(pose) @ side_view_4x4) # the inverse of extrinsic matrix
    intrinsics_all = np.stack(intrinsics_all)
    extrinsics = np.stack(poses_all)

    for i in range(int(st_num_), data_num):
        camera_dict = {}
        camera_dict["CameraReso"] = [W_, H_]
        camera_dict["ExterMat"] = np.eye(4)
        camera_dict["InterMat"] = intrinsics_all[i,...]

        render_color_mesh(meshes_path+str(i)+'.ply', scale_mats[i,...], extrinsics[i,...], camera_dict, save_path, depth_save_path, str(i))

        print(i)