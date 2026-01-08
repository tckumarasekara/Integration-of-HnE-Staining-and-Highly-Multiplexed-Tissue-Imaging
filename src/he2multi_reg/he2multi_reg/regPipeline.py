from skimage.transform import warp
from skimage.util import img_as_float32
import numpy as np
import itk
from .preprocess import load_and_scale_images, colour_deconvolusion_preprocessing_HnE, extract_channel
from .reg import register_DAPI_HnE
from .metrics import compute_TRE, compute_mutual_information

def registration_pipeline(fixed_path, moving_path, fixed_px_sz, moving_px_sz, fixed_img, adv_tform=None, feature_tform=None, intensity_tform=None,):
    """
    Registration pipeline for registering multiplexed and HnE stained tissue images. Loads and preprocesses images, performs registration, and 
    evaluates registration quality using TRE and mutual information.

    Parameters:
    - fixed_path (str) : path to the fixed image 
    - moving_path (str) : path to the moving image
    - fixed_px_sz (float or None) : pixel size of the fixed image in micrometers/pixel (if None, will attempt to read from metadata)
    - moving_px_sz (float or None) : pixel size of the moving image in micrometers/pixel (if None, will attempt to read from metadata)
    - fixed_img (str) : type of fixed image, either 'multiplexed' or 'hne'
    - adv_tform (str or None) : advanced transformation type for intensity based registration, either 'intensity', 'feature', or None   
    - feature_tform (str or None) : feature based transformation type, either 'affine', 'projective', or None
    - intensity_tform (str or None) : intensity based transformation type, either 'rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs', 'r-af', 'r-bs', or None

    Returns:
    - transformation_maps (dict) : dictionary of transformation maps (skimage Transform objects or itk Transform objects)
                            'initial similarity' : skimage Transform object for initial feature based registration
                            'intensity based' : dictionary of intensity based registration transforms (if any)
                                                'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
                                                OR
                            'affine' or 'projective' : skimage Transform object                                         
    - registered_imgs (dict) : dictionary of registered images after each registration step
                            'initial similarity' : np.array of registered image after initial feature based registration
                            'intensity based' : dictionary of registered images after each intensity based registration step (if any)
                                                'rigid' and/or 'affine' and/or 'bspline' : np.array of registered image
                                                OR  
                            'affine' or 'projective' : np.array of registered image                                       
    - moved_img (np.array) : final moved image after all registration steps
    - tre (dict or None) : dictionary of TRE values before and after each registration step
                            'before registration' : float rTRE before registration
                            'initial similarity' : float rTRE after initial feature based registration
                            'rigid' and/or 'affine' and/or 'bspline' OR 'affine' or 'projective' : float rTRE after 
                                                                                                    each registration (if any)
                            (None if TRE computation failed)
    - mi (dict or None) : dictionary of mutual information values before and after each registration step
                            'before registration' : float MI before registration
                            'initial similarity' : float MI after initial feature based registration
                            'rigid' and/or 'affine' and/or 'bspline' OR 'affine' or 'projective' : float MI after 
                                                                                                    each registration (if any)
                            (None if MI computation failed)
    """
    
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


