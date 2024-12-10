# Stage 1: Video
Convert the RGB video to images and make sparse sampling.

## Step 1: Convert an RGB video to images
```
python step1_video_to_imgs.py --path $video_path
```

## Step 2: Sample frames
```
python step2_sample_frames.py --path $video_path --gap $gap --gap2 $gap2 --split $split
```
Our video content can be roughly divided into two parts:

* Part 1: Participant keep still, while camera moving in the front.
* Part 2: Camera keep still, while participant performing different gazes.

The meanning of parameters are listed as follows:

* `gap`: sample one image for every `gap` images (Part 1)
* `gap2`: sample one image for every `gap2` images (Part 2)
* `split`: the split index for Part 1 and Part 2

In our experiments, Part 1 usually contains ~350 frames, while Part 2 usually contains ~150 frames.

## Step 3: Check and reorder (optional)
```
python step3_reorder.py --path $video_path
```
The result of Step 2 may contain some low-quality frames and need to be removed manually. After removal, the frame index need to be reordered.

# Stage 2: Camera parameters
Our method get camera parameters by [MetaShape](https://www.agisoftmetashape.com). The following steps assume you have saved the outputs of MetaShape GUI (`doc.xml` and `model.obj`) into `output` directory.

```
<path>
|-- images
|-- output
    |-- doc.xml
    |-- model.obj
...
```

## Step 1: Post-process MetaShape output
```
python step1_postprocess_metashape.py --path $path
```

## Step 2: Convert the format to COLMAP style
```
python step2_to_colmap.py --path $path
```

The coordinates of raw MetaShape output are usually messy. We prefer to make it axis-aligned, making the z-axis point to human face, the y-axis point down and the x-axis parallel with the line connecting the right and left eyes.

This can be achieved by manually providing x-axis and y-axis directions, which are saved in `sparse_raw/0/manual_info.txt` with the format as follows:
```
4.084441 -0.441589 1.115694
0.467825 7.897467 0.948918
```

Then run the script again with `--fh`.
```
python step2_to_colmap.py --path $path --fh
```

## Step 3: Process COLMAP format data
```
python step3_imgs_to_poses.py $path
```
After this step, a sparse point cloud is saved in `$path/sparse_points.ply`. Please remove the uninterested points manually and save it as `$path/sparse_points_interest.ply`.

## Step 4: Define the region of interest
```
python step4_gen_cameras.py $path
```
The script will outputs `cameras_sphere.npz`.


# Stage 3: Images
Our method use [video matting](https://github.com/PeterL1n/RobustVideoMatting) to get the face masks, and use [commerical service/software](https://www.sensetime.com/) to get the iris/eyelid landmarks. The following steps assume you have saved hard face masks in `mask` directory and iris/eyelid landmarks in `landmark` directory.

```
<path>
|-- rgb (renamed from "images")
|-- mask
|-- landmark
...
```

## Step 1: Convert landmarks to masks
```
python step1_get_eyemasks.py --dir $path --both
```

Results will be saved in the following directories:

* `irismask`: iris masks of left eye (image-right eye)
* `irismask2`: iris masks of right eye (image-left eye)
* `eyemask`: eyelid masks of left eye (image-right eye)
* `eyemask2`: eyelid masks of right eye (image-left eye)

## Step 2: Exclude eye regions to get final masks
```
python step2_exclude_eyemasks.py --dir $path --both
```

Final masks will be saved in `newmask` directory.

# Stage 4: Crop and resize (optional but recommended)
In order to avoid wasting computational resources on background pixels, it is beneficial to crop images (and modify the camera parameters accordingly).

```
# preview
python stepx_crop_resize.py --path $path --crop out_w out_h x y

# run
python stepx_crop_resize.py --path $path --crop out_w out_h x y --go
```
