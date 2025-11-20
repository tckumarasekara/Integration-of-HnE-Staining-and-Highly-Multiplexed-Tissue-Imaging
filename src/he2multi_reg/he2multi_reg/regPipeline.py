from skimage.transform import warp
from skimage.util import img_as_float32
import itk
import numpy as np
from .preprocess import load_and_scale_images, colour_deconvolusion_preprocessing_HnE, extract_channel
from .reg import register_DAPI_HnE
from .metrics import compute_TRE, compute_mutual_information

def registration_pipeline(fixed_path, moving_path, fixed_px_sz, moving_px_sz, fixed_img, adv_tform=None, feature_tform=None, intensity_tform=None,):
    
    # load and scale images 
    fixed_init, moving_init = load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz)
    print("Images loaded.")

    # preprocess HnE image
    if fixed_img == 'multiplexed':
        moving_prepr = colour_deconvolusion_preprocessing_HnE(moving_init)
        fixed_prepr = fixed_init
    elif fixed_img == 'hne':
        fixed_prepr = colour_deconvolusion_preprocessing_HnE(fixed_init)
        if len(moving_init.shape) == 2:
            moving_prepr = moving_init
        else:
            moving_prepr = extract_channel(moving_init, 0)
    else:
        raise ValueError("fixed_img must be either 'multiplexed' or 'hne'")
    print("Preprocessing completed.")
    
    
    # registration
    if len(fixed_init.shape) == 2:
        h, w = fixed_init.shape
    else:
        h, w, c = fixed_init.shape

    transformation_maps, registered_imgs, tre_pts = register_DAPI_HnE(fixed_prepr, moving_prepr, adv_tform, feature_tform, intensity_tform, mpp=moving_px_sz)
    moved_init = warp(moving_init, transformation_maps['initial similarity'].inverse, output_shape=(h, w, moving_init.shape[2]) if len(moving_init.shape) == 3 else (h, w))

    if adv_tform is None:
        moved_img = moved_init

    if adv_tform == 'intensity':

        if len(moving_init.shape) == 2:
            result = itk.GetImageFromArray(img_as_float32(moved_init))
            result.SetSpacing([moving_px_sz, moving_px_sz]) 
       
            for key, value in transformation_maps['intensity based'].items():
                result = itk.transformix_filter(result, value, log_to_console=False)
        
            moved_img = itk.GetArrayFromImage(result)

        else:
            channels = []
            
            for c in range(moving_init.shape[2]):
                img = moved_init[:,:,c] 
                itk_img = itk.GetImageFromArray(img_as_float32(img))
                itk_img.SetSpacing([moving_px_sz, moving_px_sz])

                for key, value in transformation_maps['intensity based'].items():
                    itk_img = itk.transformix_filter(itk_img, value, log_to_console=False)
        
                transformed_channel = itk.GetArrayFromImage(itk_img)
                channels.append(transformed_channel)

            min_h = min(ch.shape[0] for ch in channels)
            min_w = min(ch.shape[1] for ch in channels)
            channels_cropped = [ch[:min_h, :min_w] for ch in channels]
            moved_img = np.stack(channels_cropped, axis=-1)

            moved_img = np.clip(moved_img, 0, 1)
            
    elif adv_tform == 'feature':
        key = list(transformation_maps.keys())[1]
        moved_img = warp(moving_init, transformation_maps[key].inverse, output_shape=(h, w, moving_init.shape[2]) if len(moving_init.shape) == 3 else (h, w))


    # evaluate registration with metrics
    try:
        tre = compute_TRE(transformation_maps, tre_pts, fixed_prepr, mpp=moving_px_sz)
    except ValueError as e:
        print("TRE computation skipped:", e)
        tre = None  
    except Exception as e:
        print("An unexpected error occurred during TRE computation:", e)
        tre = None

    try:
        mi = compute_mutual_information(fixed_prepr, moving_prepr, registered_imgs)
    except Exception as e:
        print("An unexpected error occurred during mutual information computation:", e)
        mi = None

    return transformation_maps, registered_imgs, moved_img, tre, mi


