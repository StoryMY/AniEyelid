import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def draw_gaze_scatter(gaze, save_path, split=350):
    plt.cla()
    color = np.zeros([gaze.shape[0], 3])
    color[:split] = np.array([0, 0, 1])
    plt.scatter(gaze[:, 1], gaze[:, 0], c=color)
    plt.scatter(gaze[:split, 1], gaze[:split, 0], c=color[:split])
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig(save_path)

def draw_gaze_scatter_seq(gaze, save_path, split=350):
    os.makedirs(save_path, exist_ok=True)
    color = np.zeros([gaze.shape[0], 3])
    color[:split] = np.array([0, 0, 1])
    plt.figure(figsize=(4, 4), dpi=100)
    for i in tqdm(range(gaze.shape[0])):
        plt.cla()
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)
        plt.scatter(gaze[i, 1], gaze[i, 0], c=color[i])
        # plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(save_path, '%03d.png' % i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results')
    parser.add_argument('--split', type=int, required=True, help='same with the split in config files for initialization')
    args = parser.parse_args()

    parameters = np.load(os.path.join(args.dir, 'parameters_mergesplit.npz'))
    gaze1 = parameters['gaze']
    gaze2 = parameters['gaze2']
    draw_gaze_scatter(gaze1, os.path.join(args.dir, 'gaze1.png'), args.split)
    draw_gaze_scatter(gaze2, os.path.join(args.dir, 'gaze2.png'), args.split)

    # Post-Process
    gaze_mean1 = np.mean(gaze1[:args.split], axis=0)
    gaze_new1 = gaze1.copy()
    gaze_new1[:args.split] = gaze_mean1
    print(gaze_mean1)
    draw_gaze_scatter(gaze_new1, os.path.join(args.dir, 'gaze1_new.png'), args.split)

    gaze_mean2 = np.mean(gaze2[:args.split], axis=0)
    gaze_new2 = gaze2.copy()
    gaze_new2[:args.split] = gaze_mean2
    print(gaze_mean2)
    draw_gaze_scatter(gaze_new2, os.path.join(args.dir, 'gaze2_new.png'), args.split)

    np.savez(os.path.join(args.dir, 'parameters_mergesplit_new.npz'),
                gaze=gaze_new1,
                gaze2=gaze_new2)

