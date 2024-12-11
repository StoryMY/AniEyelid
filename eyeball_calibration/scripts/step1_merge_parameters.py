import os
import numpy as np
import argparse

def merge_sample_points(d1, d2, save_dir, fname):
    print('merge', fname)
    sp1 = np.load(os.path.join(d1, fname), allow_pickle=True)
    sp2 = np.load(os.path.join(d2, fname), allow_pickle=True)

    n_data = len(sp1['xyz'])

    xyz = [np.concatenate([sp1['xyz'][i], sp2['xyz'][i]]) for i in range(n_data)]
    nor = [np.concatenate([sp1['nor'][i], sp2['nor'][i]]) for i in range(n_data)]
    xyz_half = [np.concatenate([sp1['xyz_half'][i], sp2['xyz_half'][i]]) for i in range(n_data)]

    xyz = np.array(xyz, dtype=object)
    nor = np.array(nor, dtype=object)
    xyz_half = np.array(xyz_half, dtype=object)

    print(n_data, len(xyz))

    np.savez(os.path.join(save_dir, fname[:-4] + '_merge.npz'), xyz=xyz, nor=nor, xyz_half=xyz_half)


def merge_gaze(d1, d2, save_dir, fname):
    print('merge', fname)
    g1 = np.load(os.path.join(d1, fname))
    g2 = np.load(os.path.join(d2, fname))

    gaze = (g1['gaze'] + g2['gaze']) * 0.5

    np.savez(os.path.join(save_dir, fname[:-4] + '_merge.npz'), 
             gaze=gaze,
             translate=g1['translate'],
             scale=g1['scale'],
             translate2=g2['translate'],
             scale2=g2['scale'])
    
    
def merge_gaze_split(d1, d2, save_dir, fname):
    print('merge', fname)
    g1 = np.load(os.path.join(d1, fname))
    g2 = np.load(os.path.join(d2, fname))

    np.savez(os.path.join(save_dir, fname[:-4] + '_mergesplit.npz'), 
             gaze=g1['gaze'],
             gaze2=g2['gaze'],
             translate=g1['translate'],
             scale=g1['scale'],
             translate2=g2['translate'],
             scale2=g2['scale'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d1', type=str, required=True, help='path to directory 1')
    parser.add_argument('--d2', type=str, required=True, help='path to directory 2')
    parser.add_argument('--out', type=str, default='results', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    merge_sample_points(args.d1, args.d2, args.out, 'sample_points.npz')
    merge_gaze(args.d1, args.d2, args.out, 'parameters.npz')
    merge_gaze_split(args.d1, args.d2, args.out, 'parameters.npz')
