# Eyeball Calibration
Given images and camera parameters, this repository outputs eyeball shapes (position + uniform scale) and motions (rotations).

# Installation

**Notes:** The gcc version should <= 10.0

## Step 1: Create Conda Environment
```
# python=3.9 have been tested
conda create -n eyecalib python=3.9
conda activate eyecalib
```

## Step 2: Install Pytorch
Find the suitable version from [Previous Pytorch Versions](https://pytorch.org/get-started/previous-versions/).
```
# pytorch==1.11.0 with cudatoolkit==11.3 has been tested
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## Step 3: Install Packages of Differentiable Rendering
Details can be found in [nvdiffrast](https://github.com/NVlabs/nvdiffrast), [nvdiffrec](https://github.com/NVlabs/nvdiffrec) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
```
pip install ninja imageio PyOpenGL glfw xatlas gdown

# install nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

## Step 4: Install Other Tool Packages
```
pip install -r requirements.txt
```


# Calibration
We calibrate left eye (OS) and right eye (OD) separately.

Before run calibration scripts, we need first modify configs.

### init_xx.yaml
* `root_dir`: data directory
* `select`: the frame index used for initialization, we recommend to select about 10 representative images (e.g., camera in left/right/up/down and gaze to left/right/up/down)
* `split`: the split index for Part 1 and Part 2 (Part 1 frames will share the same gaze)

### optim_xx.yaml
* `root_dir`: data directory
* `select`: keep it as an empty list, which means all frames will join in optimization.
* `split`: keep it as -1

Then, run the main script with `GPU 0`:
```
bash run.sh 0
```

# Post-Process
## Step 1: Merge parameters of left and right eyes
```
python scripts/step1_merge_parameters.py --d1 ${save_dir}/${case}_os --d2 ${save_dir}/${case}_od
```
The results will be saved in `results` directory by default.

## Step 2: Average gazes of Part 1 
```
python scripts/step2_postproc_gaze.py --split $split        # coordinated gaze of left/right eyes
python scripts/step2_postproc_gazesplit.py --split $split   # separate gaze of left/right eyes
```

## Step 3: Move all .npz files to the data directory
```
mv results/*.npz $path
```

Finally, the data directory is organized as follows:
```
<path>
|-- rgb
|-- newmask
|-- sparse
|-- *.npz
...
```

