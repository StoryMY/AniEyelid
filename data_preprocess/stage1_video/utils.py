import os

def smart_mkdir(video_path, out_dir=None):
    if out_dir is None:
        out_dir = os.path.split(video_path)[0]
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    work_dir = os.path.join(out_dir, video_name)
    image_dir = os.path.join(work_dir, 'images_raw')
    sparse_dir = os.path.join(work_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    return image_dir, sparse_dir, work_dir
