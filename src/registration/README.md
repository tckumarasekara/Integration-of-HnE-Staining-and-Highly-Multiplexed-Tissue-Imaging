# he2multi-reg

**he2multi-reg** is a command-line tool and a Python package for registering H&E stained and multiplexed tissue images. It provides a feature and intensity-based registration pipeline, saving registered images, transformation maps and evaluation metrics.


## Features

- Register H&E images and multiplexed images (after extracting DAPI channel) using transformations.
- Supports feature-based and intensity-based registration.
- Outputs registered images (in the pixel size of moving image), transformation maps and evaluation metrics (TRE and Mutual Information).


## Recommendations

- For most cases, it is recommended to register **H&E images onto multiplexed images** (H&E as moving image).  
- The **initial registration without any advanced transformation** usually works well and faster without any advanced transformations.


## Installation

You can install **he2multi-reg** using pip:

```bash
pip install he2multi_reg
```

---

## Usage as a command-line tool

### Register Images

```bash
he2multi-reg register <fixed_path> <moving_path> <output_folder> <fixed_img> [options]
```

#### Examples:

```bash
he2multi-reg register data/fixed_img.ome.tiff data/moving_img.ome.tiff ../output multiplexed
```
```bash
he2multi-reg register data/fixed_img.tif data/moving_img.tif ../output multiplexed --fixed-px-sz 0.21 --moving-px-sz 0.52
```

#### Arguments:

- **fixed_path**: Path to the fixed image (H&E or DAPI or Multiplexed image path)
- **moving_path**: Path to the moving image (H&E or DAPI or Multiplexed image path)
- **output_folder**: Folder to save the registered images and metrics  
- **fixed_img**: Type of fixed image: `multiplexed` or `hne`  

#### Options:

- `--fixed-px-sz` — Pixel size of the fixed image (no need to provide for ome.tiff, so default: None)
- `--moving-px-sz` — Pixel size of the moving image (no need to provide for ome.tiff, so default: None)
- `--adv-tform` — Advanced transformation type: `feature` or `intensity` (only if advanced transformation required)
- `--feature-tform` — Feature transformation method: `affine` or `projective` (only if adv-tform is `feature`)
- `--intensity-tform` — Intensity transformation method: `rigid`, `affine`, `bspline`, etc. (only if adv-tform is `intensity`) 
- `--intermediate-imgs / --no-intermediate-imgs` — Save intermediate images (default: False)  

#### Output

After running registration, the following files/folders will be generated:

- **results/registration_metrics.json** — TRE and Mutual Information  
- **results/registered_images/** — Registered images  
- **results/transformation_maps/** — Transformation maps (.npy files for feature based registration steps and .txt files for intensity based registration steps)


### Extract a Channel (DAPI can be extracted for registration)

```bash
he2multi-reg extract-channel <file_path> <output_folder_path> [--channel-idx N]
```

- `--channel-idx`: Channel index to extract (default: 0 for DAPI)
 
#### Output

- **/multiplexed_channel_{channel_idx}.tif** - Image with the extracted channel

---

## Usage as a Python Library

Although **he2multi-reg** is mainly a command-line tool, its functions can also be used directly in Python for scripting.

### Example: Running the Registration Pipeline

```python
from he2multi_reg.regPipeline import registration_pipeline

# run registration pipeline
tform_maps, registered_imgs, final_img, tre, mi = registration_pipeline(
    fixed_path="fixed_image.tif",
    moving_path="moving_image.tif",
    fixed_px_sz=0.5,
    moving_px_sz=0.5,
    fixed_img="multiplexed",
    adv_tform="intensity",        # or "feature"
    feature_tform="affine",       # used if adv_tform="feature"
    intensity_tform="rigid"       # used if adv_tform="intensity"
)

print("TRE:", tre)
print("Mutual Information:", mi)
```
---

## License

This project is licensed under the **MIT License**.




