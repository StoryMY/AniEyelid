import numpy as np
import torch

def gen_gaze(pitch_max, pitch_num, yaw_max, yaw_num):
    pitch = np.linspace(-1, 1, pitch_num) * pitch_max
    yaw = np.linspace(-1, 1, yaw_num) * yaw_max

    col, row = np.meshgrid(yaw, pitch)
    gaze = np.stack([row, col], axis=2).reshape([-1, 2])

    assert gaze.shape[0] == pitch_num * yaw_num

    return gaze

def gen_circle_gaze(pitch_max, yaw_max, gaze_num):
    pitch = np.linspace(-1, 1, gaze_num) * pitch_max
    yaw = np.linspace(-1, 1, gaze_num) * yaw_max
    pitch_inv = pitch[::-1]
    yaw_inv = yaw[::-1]
    pitch = np.concatenate([pitch[:-1], pitch_inv[:-1]])
    yaw = np.concatenate([yaw[:-1], yaw_inv[:-1]])

    gaze = []
    p_st = gaze_num // 2
    y_st = 0
    for i in range(2*gaze_num-2):
        p_idx = (p_st + i) % (2*gaze_num-2)
        y_idx = (y_st + i) % (2*gaze_num-2)

        temp = [pitch[p_idx], yaw[y_idx]]
        gaze.append(temp)

    return np.array(gaze)
