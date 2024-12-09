import cv2
import os
import json
import numpy as np
import argparse
import trimesh
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from colmap.read_write_model import read_cameras_binary, read_points3d_binary, read_images_binary, rotmat2qvec, write_images_binary, write_cameras_binary, write_points3d_binary
import colmap.read_write_model as rwm

def proj_3d_to_2d(p3d, cam_mat):
    """
        :param p3d: 3d points [n, 3]
        :return: 2d points [n, 2]
    """
    p2d_raw = (cam_mat @ p3d.T).T
    p2d = p2d_raw / p2d_raw[:, 2:3]

    return p2d[:, :2]

def read_model(path, ext=".bin"):
    cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
    images = read_images_binary(os.path.join(path, "images" + ext))
    points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def metashape_to_colmap(args, front_mat=None):
    # mkdir
    if front_mat is None:
        save_dir = os.path.join(args.path, 'sparse_raw', '0')
    else:
        save_dir = os.path.join(args.path, 'sparse', '0')
    os.makedirs(save_dir, exist_ok=True)

    # load metashape
    with open(os.path.join(args.path, 'transforms.json'), 'r') as fjson:
        cam_obj = json.load(fjson)

    # camera matrix
    fx = cam_obj['fl_x']
    cx = cam_obj['cx']
    cy = cam_obj['cy']
    new_cam_item = rwm.Camera(id=1, model='SIMPLE_PINHOLE', 
                        width=int(cam_obj['w']), height=int(cam_obj['h']),
                        params=np.array([fx, cx, cy]))
    new_cam = {}
    new_cam[1] = new_cam_item
    write_cameras_binary(new_cam, os.path.join(save_dir, 'cameras.bin'))

    # camera pose
    new_img = {}
    total_frame = len(cam_obj['frames'])
    print('[!!!] Check this number:', total_frame)  # metashape may skip some frames
    for fid in range(total_frame):
        transmat = np.array(cam_obj['frames'][fid]['transform_matrix'])
        if front_mat is not None:
            transmat = front_mat @ transmat
        transmat_inv = np.linalg.inv(transmat)
        rmat = transmat_inv[:3, :3]
        tvec = transmat_inv[:3, 3].reshape(3)
        qvec = rotmat2qvec(rmat)
        filename = os.path.split(cam_obj['frames'][fid]['file_path'])[-1].strip()
        # init
        new_img_item = rwm.Image(id=fid+1, qvec=qvec, tvec=tvec,
                                camera_id=1, name=filename, 
                                xys=np.array([[1, 2]]), point3D_ids=np.array([1])) # dummy params
        new_img[fid+1] = new_img_item
    write_images_binary(new_img, os.path.join(save_dir, 'images.bin'))

    # point cloud
    scene = trimesh.load_mesh(os.path.join(args.path, 'output', 'model.obj'))
    # mesh_list = [scene.geometry[key] for key in scene.geometry.keys()]
    mesh_list = [scene]

    convert_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    if front_mat is not None:
        convert_mat = front_mat @ convert_mat
    
    points_merge = []
    colors_merge = []
    for mesh in mesh_list:
        # points 3d
        points = mesh.vertices
        points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        points = (points @ convert_mat.T)[:, :3]
        # mesh_color = mesh.visual.to_color()
        # colors = mesh_color.vertex_colors[:, :3]
        colors = mesh.visual.vertex_colors[:, :3]

        points_merge.append(points)
        colors_merge.append(colors)
    points_merge = np.concatenate(points_merge)
    colors_merge = np.concatenate(colors_merge)

    new_points = {}
    for i, pts in enumerate(points_merge):
        new_points_item = rwm.Point3D(id=i+1, xyz=pts, rgb=colors_merge[i], 
                                    error=np.array(0.0), image_ids=np.array([1, 2, 3]), 
                                    point2D_idxs=np.array([1, 2, 3]))   # dummy params
        new_points[i+1] = new_points_item
    write_points3d_binary(new_points, os.path.join(save_dir, 'points3D.bin'))
    pcd = trimesh.PointCloud(points_merge, colors_merge)
    pcd.export(os.path.join(save_dir, 'object_point_cloud.ply'))

    # # check read
    # read_model(save_dir)

    print('Done!')

def get_front_mat(args):
    save_dir = os.path.join(args.path, 'sparse_raw', '0')
    pcd_path = os.path.join(save_dir, 'object_point_cloud.ply')
    manual_info_path = os.path.join(save_dir, 'manual_info.txt')
    # read
    x = None
    y = None
    with open(manual_info_path, 'r') as f:
        lines = f.readlines()
        first_row = lines[0].strip().split()
        second_row = lines[1].strip().split()
        x = np.array([float(item) for item in first_row])
        y = np.array([float(item) for item in second_row])
    
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)
    z = np.cross(x_norm, y_norm)
    z_norm = z / np.linalg.norm(z)
    y_new = np.cross(z_norm, x_norm)        # [Important] fix bug
    y_norm = y_new / np.linalg.norm(y_new)


    R_inv = np.concatenate([x_norm.reshape([3, 1]), y_norm.reshape([3, 1]), z_norm.reshape([3, 1])], axis=1)
    R = R_inv.T
    rotate_mat = np.eye(4, 4)
    rotate_mat[:3, :3] = R

    # load point cloud
    pcd = trimesh.load_mesh(pcd_path)
    t = -np.mean(pcd.vertices, axis=0)
    trans_mat = np.eye(4, 4)
    trans_mat[:3, 3] = t
    trans_mat_inv = np.eye(4, 4)
    trans_mat_inv[:3, 3] = -t

    front_mat = trans_mat_inv @ rotate_mat @ trans_mat
    print('[INFO] front head matrix')
    print(front_mat)
    return front_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--fh', action='store_true', help='Make front head')
    args = parser.parse_args()
    print(args)

    if not args.fh:         # step 1
        metashape_to_colmap(args, None)
    else:                   # step 2
        front_mat = get_front_mat(args)
        metashape_to_colmap(args, front_mat)
