import numpy as np
import trimesh
import cv2 as cv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from glob import glob
import colmap.read_write_model as read_model

def get_bound_sphere(center, radius):
    phi_arr = np.linspace(0, np.pi*2, 20)
    theta_arr = np.linspace(0, np.pi, 10)

    points = []
    for phi in phi_arr:
        for theta in theta_arr:
            x = np.sin(theta) * np.cos(phi) * radius
            y = np.sin(theta) * np.sin(phi) * radius
            z = np.cos(theta) * radius
            points.append([x, y, z])
    points = np.array(points) + center
    return points

def get_cx_cy(work_dir):
    camerasfile = os.path.join(work_dir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print(cam.params)

    return cam.params[1], cam.params[2]

if __name__ == '__main__':
    work_dir = sys.argv[1]
    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))
    #
    cx, cy = get_cx_cy(work_dir)

    cam_dict = dict()
    n_images = len(poses_raw)
    print(n_images)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = cx # (w - 1) * 0.5
        intrinsic[1, 2] = cy # (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)


    pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    # radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()        # loose bbox
    radius = np.abs(vertices - center).max() * 1.00                         # tight bbox
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center
    sphere_points = get_bound_sphere(center, radius)
    sphere = trimesh.PointCloud(vertices=sphere_points)
    sphere.export(os.path.join(work_dir, 'sphere.ply'))

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    # out_dir = os.path.join(work_dir, 'preprocessed')
    # os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    # os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    # image_list = glob(os.path.join(work_dir, 'images/*.png'))
    # # image_list = glob(os.path.join(work_dir, 'images/*.jpg'))
    # image_list.sort()

    # for i, image_path in enumerate(image_list):
    #     img = cv.imread(image_path)
    #     cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
    #     cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

    # np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    np.savez(os.path.join(work_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')
