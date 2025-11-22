import histomicstk as htk
import numpy as np
import os
from tifffile import imread, TiffFile
from skimage.transform import resize
import xml.etree.ElementTree as ET



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



def get_image_size_ome_tiff(file_path):

    with TiffFile(file_path) as tif:
        img = tif.series[0].asarray()
        shape = img.shape[0:2] if img.ndim == 2 or img.shape[0] < img.shape[2] else img.shape[1:3] 
        return shape



def get_pixel_size_ome_tiff(file_path):
    with TiffFile(file_path) as tif:
        ome = tif.ome_metadata
        if ome is None:
            raise ValueError(f"Not an OME-TIFF: {file_path}")

        root = ET.fromstring(ome)
        pixels = root.find(".//{*}Pixels")   

        px = pixels.get("PhysicalSizeX")
        py = pixels.get("PhysicalSizeY")

        px = float(px) if px is not None else None
        py = float(py) if py is not None else None

        return px, py



def load_image_data(file_path):
    if file_path.endswith(".tif") or file_path.endswith(".tiff"):
        img_raw = imread(file_path)
        img = np.array(img_raw) 

        return img if (len(img.shape) == 2) or (img.shape[2] < img.shape[0]) else img.transpose(1, 2, 0)
    
    else: 
        raise ValueError("Unsupported file format. Please provide a .tif file.")



def extract_channel(img, channel_index):
    
    return img[:, :, channel_index]



def load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz):

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")

    if moving_px_sz is None:
        try:
            moving_px_sz, _ = get_pixel_size_ome_tiff(moving_path)
        except Exception:
            moving_px_sz = None

        if moving_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for moving image. Please provide moving_px_sz.")

    scale = moving_px_sz / fixed_px_sz

    # load fixed image
    fixed_img = load_image_data(fixed_path)
    if len(fixed_img.shape) == 2:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale)), anti_aliasing=True)
    elif fixed_img.shape[2] == 3:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale), fixed_img.shape[2]), anti_aliasing=True)
    elif fixed_img.shape[2] > 3:
        fixed_ch_img = extract_channel(fixed_img, 0)
        fixed_init = resize(fixed_ch_img, (int(fixed_ch_img.shape[0]/scale), int(fixed_ch_img.shape[1]/scale)), anti_aliasing=True)
    fixed_init = fixed_init*255

    # load moving image
    moving_init = load_image_data(moving_path)

    return fixed_init, moving_init
