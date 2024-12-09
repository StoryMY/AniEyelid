import os
import argparse
from utils import smart_mkdir

def main(args):
    image_dir, sparse_dir, _ = smart_mkdir(args.path)

    imglist = os.listdir(sparse_dir)
    imglist.sort()

    ctr = 0
    for i, imgname in enumerate(imglist):
        in_path = os.path.join(sparse_dir, imgname)
        out_path = os.path.join(sparse_dir, '%03d.png' % ctr)
        os.rename(in_path, out_path)
        ctr += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='The video path')
    args = parser.parse_args()
    print(args)

    main(args)
