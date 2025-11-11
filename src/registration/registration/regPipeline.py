from .preprocess import load_and_scale_images, colour_deconvolusion_preprocessing_HnE
from .reg import register_DAPI_HnE
from .metrics import compute_TRE, compute_mutual_information

def registration_pipeline(fixed_path, moving_path, fixed_px_sz, moving_px_sz, fixed_img, adv_tform=None, feature_tform=None, intensity_tform=None,):
    
    # load and scale images 
    fixed_init, moving_init = load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz)

    # preprocess HnE image
    if fixed_img == 'multiplexed':
        moving_prepr = colour_deconvolusion_preprocessing_HnE(moving_init)
        fixed_prepr = fixed_init
    elif fixed_img == 'hne':
        fixed_prepr = colour_deconvolusion_preprocessing_HnE(fixed_init)
        moving_prepr = moving_init
    else:
        raise ValueError("fixed_img must be either 'multiplexed' or 'hne'")
    
    # registration
    transformation_maps, registered_imgs, tre_pts = register_DAPI_HnE(fixed_prepr, moving_prepr, adv_tform, feature_tform, intensity_tform, mpp=moving_px_sz)

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

    return transformation_maps, registered_imgs, tre, mi


