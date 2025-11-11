import histomicstk as htk
import numpy as np
import os
from tifffile import imread
from skimage.transform import resize



def colour_deconvolusion_preprocessing_HnE(hne_init):
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains

    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(hne_init, W)
    hne_deconv = 1 - imDeconvolved.Stains[:, :, 0]

    return hne_deconv



def load_image_data(file_path):
    if file_path.endswith(".tif"): 
        img_raw = imread(file_path)
        img = np.array(img_raw)

        return img
    
    else: 
        raise ValueError("Unsupported file format. Please provide a .tif file.")



def load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz):
    scale = moving_px_sz / fixed_px_sz

    # load fixed image
    fixed_img = load_image_data(fixed_path)
    fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale)), anti_aliasing=True)
    fixed_init = fixed_init*255

    # load moving image
    moving_init = load_image_data(moving_path)

    return fixed_init, moving_init
