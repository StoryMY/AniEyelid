import cv2
import numpy as np

def read_lmk(file):
    face_lmk_all = []
    face_lmk = []
    eyelid_lmk = []
    iris_lmk = []
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return face_lmk, eyelid_lmk, iris_lmk
        face_lmk_num = int(lines[0])
        face_strs = lines[1].split()
        for i in range(face_lmk_num):
            x = float(face_strs[2 * i])
            y = float(face_strs[2 * i + 1])
            face_lmk_all.append([x, y])

        eyelid_lmk_num = int(lines[2])
        eyelid_strs = lines[3].split()
        for i in range(eyelid_lmk_num):
            x = float(eyelid_strs[2 * i])
            y = float(eyelid_strs[2 * i + 1])
            eyelid_lmk.append([x, y])

        iris_lmk_num = int(lines[4])
        iris_strs = lines[5].split()
        for i in range(iris_lmk_num):
            x = float(iris_strs[2 * i])
            y = float(iris_strs[2 * i + 1])
            iris_lmk.append([x, y])

        face_lmk += face_lmk_all[0:33:2]
        face_lmk += face_lmk_all[33:64]
        face_lmk += face_lmk_all[84:104]
    return np.array(face_lmk), np.array(eyelid_lmk), np.array(iris_lmk)


def generate_seg(img, lmk, is_eyelid=False):
    h, w = img.shape[:2]
    seg = np.zeros([h, w], dtype=np.uint8)
    new_lmk = (lmk + 0.5).astype(np.int32)
    cv2.fillPoly(seg, [new_lmk], color=(255, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    seg = cv2.dilate(seg, kernel)
    seg = cv2.erode(seg, kernel)
    return seg
