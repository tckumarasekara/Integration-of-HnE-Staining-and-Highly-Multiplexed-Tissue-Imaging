from pathlib import Path
import random
import numpy as np
from skimage.util import img_as_float32
from skimage.transform import resize, estimate_transform, warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage import measure
import itk


def transform_seg_mask(mask, transformation_maps, output_shape, mpp=None):
    """
    Transform segmentation mask using the provided transformation maps.

    Parameters:
    - mask (np.array) : segmentation mask to be transformed
    - transformation_maps (dict) : dictionary of transformation maps (skimage Transform objects or itk Transform objects)
                                    'initial similarity' : skimage Transform object for initial feature based registration
                                    'intensity based' : dictionary of intensity based registration transforms (if any)
                                                        'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
                                                        OR
                                    'affine' or 'projective' : skimage Transform object
    - output_shape (tuple) : desired output shape of the transformed mask (shape of the fixed image)
    - mpp (float, optional) : pixel size only required for transforming segmentation mask after intensity based registration

    Returns:
    - moved_mask (np.array) : transformed segmentation mask
    """

    moved_mask = warp(mask, transformation_maps['initial similarity'].inverse, output_shape=output_shape, order=0, preserve_range=True)

    if 'intensity based' in transformation_maps:
        if mpp is None:
            raise ValueError("mpp must be provided for transforming segmentation mask after intensity based registration")

        for _, tform in transformation_maps['intensity based'].items():

            moving_mask = itk.GetImageFromArray(moved_mask.astype(np.float32))
            moving_mask.SetSpacing([mpp,mpp])

            tform.SetParameter("FinalBSplineInterpolationOrder", '0')

            moving_mask = itk.transformix_filter(
                moving_mask, 
                tform,
                log_to_console=False
            )

            moved_mask = itk.GetArrayFromImage(moving_mask)
            min_val, max_val = mask.min(), mask.max()
            moved_mask = np.clip(np.rint(moved_mask), min_val, max_val).astype(mask.dtype)

            
    elif 'intensity based' not in transformation_maps and len(transformation_maps) > 1:
        key = list(transformation_maps.keys())[1]
        moved_mask = warp(mask, transformation_maps[key].inverse, output_shape=output_shape, order=0, preserve_range=True)

    return moved_mask



def apply_tform_adv(moving_img_itk, transform_map):
    """
    Apply advanced transformations to the moving image using the provided transformation maps.

    Parameters:
    - moving_img_itk (itk.Image) : moving image in itk format   
    - transform_map (dict of itk.Transform) : dictionary of intensity based registration transforms
                                            'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
    
    Returns:
    - transformed_img (dict of np.array) : dictionary of transformed images after applying each transformation
                                        'rigid' and/or 'affine' and/or 'bspline' : np.array of transformed image
    """

    transformed_img = {}
    result = moving_img_itk

    for key, value in transform_map.items():
        result = itk.transformix_filter(result, value, log_to_console=False)
        transformed_img[key] = itk.GetArrayFromImage(result)

    return transformed_img



def register_adv_intensity_based(fixed, moving, mpp, transformation):
    """
    Register moving image to fixed image using advanced intensity based registration.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - mpp (float) : pixel size in micrometers/pixel
    - transformation (str) : type of transformation to apply ('rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs', 'r-af', 'r-bs')

    Returns:
    - global_trf_map (dict of itk.Transform) : dictionary of transformation maps after registration
                                            'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
    - registered_imgs (dict of np.array) : dictionary of registered images after applying each transformation
                                        'rigid' and/or 'affine' and/or 'bspline' : np.array of registered image
    """
    
    if transformation == 'rigid':
        transform_scheme, transform_names = ["01_Rigid"], ["rigid"]
    elif transformation == 'affine':
        transform_scheme, transform_names = ["02_Affine"], ["affine"]
    elif transformation == 'bspline':
        transform_scheme, transform_names = ["03_BSpline"], ["bspline"]
    elif transformation == 'r-af-bs':
        transform_scheme, transform_names = ["01_Rigid", "02_Affine", "03_BSpline"], ["rigid", "affine", "bspline"]
    elif transformation == 'af-bs':
        transform_scheme, transform_names = ["02_Affine", "03_BSpline"], ["affine", "bspline"]
    elif transformation == 'r-af':
        transform_scheme, transform_names = ["01_Rigid", "02_Affine"], ["rigid", "affine"]
    elif transformation == 'r-bs':
        transform_scheme, transform_names = ["01_Rigid", "03_BSpline"], ["rigid", "bspline"]
    else:
        raise ValueError("transformation for intensity based registration must be one of 'rigid', 'affine', 'bspline', 'r-af-bs', or 'af-bs'.")

    parameter_maps = []
    base_dir = Path(__file__).parent / "tform_params_adv"
    for tform in transform_scheme:
        p = base_dir / f"{tform}.txt"
        parameter_maps.append(str(p))

    fixed_itk = itk.GetImageFromArray(img_as_float32(fixed))
    fixed_itk.SetSpacing([mpp,mpp])

    moving_itk = itk.GetImageFromArray(img_as_float32(moving))
    moving_itk.SetSpacing([mpp,mpp])

    global_trf_map = {}
    for Reg, tfname in zip(parameter_maps, transform_names):
        reg_map = itk.ParameterObject.New()
        reg_map.AddParameterFile(str(Reg))

        moving_itk, result_trf_params = itk.elastix_registration_method(
            fixed_itk, 
            moving_itk,
            parameter_object = reg_map,
            log_to_console=False
        )

        global_trf_map[tfname] = result_trf_params

    registered_imgs = apply_tform_adv(moving_itk, global_trf_map)

    return global_trf_map, registered_imgs



def features_with_SIFT(fixed, moving, max_ratio=0.6, n_octaves=3, n_scales=5, scale_factor=4):
    """
    Extract features from fixed and moving images using SIFT and find matching points. Downscale images for feature extraction if 
    necessary.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - max_ratio (float) : maximum ratio for descriptor matching
    - n_octaves (int) : number of octaves for SIFT
    - n_scales (int) : number of scales per octave for SIFT
    - scale_factor (int) : factor to downscale images for feature extraction

    Returns:
    - moving_matches (np.array) : matched keypoints from moving image
    - fixed_matches (np.array) : matched keypoints from fixed image
    """

    fixedX, fixedY = fixed.shape
    movingX, movingY = moving.shape
    scale_factor = scale_factor

    if fixedX > 2000 or fixedY > 2000:
        scale_factor = max(fixedX // 2000, fixedY // 2000) * scale_factor

    elif fixedX < 250 and fixedY < 250:
        scale_factor = 1

    # Resize the images to reduce memory usage
    fixed_scaled = resize(fixed, (fixedX // scale_factor, fixedY // scale_factor), anti_aliasing=True)
    moving_scaled = resize(moving, (movingX // scale_factor, movingY // scale_factor), anti_aliasing=True)

    descriptor_extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales)

    descriptor_extractor.detect_and_extract(moving_scaled)
    keypoints1, descriptors1 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(fixed_scaled)
    keypoints2, descriptors2 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    matches12 = match_descriptors(
        descriptors1, descriptors2, max_ratio=max_ratio, cross_check=True
    )

    if matches12.shape[0] < 3:
        raise ValueError("Not enough matching points found between images for reliable registration.")

    # Extract matched keypoints
    src, dst = keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]

    dst, src = dst * scale_factor, src * scale_factor

    # Compute inliers using RANSAC 
    _, inliers = measure.ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)
    
    movingtemp_matches = src[inliers] 
    fixedtemp_matches = dst[inliers] 
    
    moving_matches = movingtemp_matches[:, [1, 0]].copy()
    fixed_matches = fixedtemp_matches[:, [1, 0]].copy()

    return [moving_matches, fixed_matches]



def register_feature_based(fixed, moving, transform_type, moving_matches, fixed_matches):
    """
    Register moving image to fixed image using feature based registration.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - transform_type (str) : type of transformation to estimate ('similarity', 'affine', 'projective')
    - moving_matches (np.array) : matched keypoints from moving image
    - fixed_matches (np.array) : matched keypoints from fixed image

    Returns:
    - tform (skimage Transform object) : estimated transformation
    - aligned_moving (np.array) : moving image aligned to fixed image
    """

    tform = estimate_transform(transform_type, src=moving_matches, dst=fixed_matches)
    aligned_moving = warp(moving, tform.inverse, output_shape=fixed.shape)

    return tform, aligned_moving


def register_init_feature_based(fixed, moving, max_ratio=0.6, n_octaves=3, n_scales=5, scale_factor=4):
    """
    Call SIFT feature detection and register moving image to fixed image using initial feature based registration. And select 
    points for TRE computation.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - max_ratio (float) : maximum ratio for descriptor matching
    - n_octaves (int) : number of octaves for SIFT
    - n_scales (int) : number of scales per octave for SIFT
    - scale_factor (int) : factor to downscale images for feature extraction

    Returns:
    - tform (skimage Transform object) : estimated transformation
    - aligned_moving (np.array) : moving image aligned to fixed image
    - tre_points (list of np.array) : points selected for TRE computation [moving_points, fixed_points]
    - reg_points (list of np.array) : points used for registration [moving_points, fixed_points]
    """

    [moving_matches, fixed_matches] = features_with_SIFT(fixed, moving, max_ratio=max_ratio, n_octaves=n_octaves, n_scales=n_scales, scale_factor=scale_factor)

    num_matches = moving_matches.shape[0]

    if num_matches < 3:
        raise ValueError(f"At least three matching points are required for initial feature based registration, only {num_matches} found.")
    
    num_tre_points = min(6, num_matches - 3, num_matches // 2)
    all_idx = set(range(num_matches))
    tre_idx = random.sample(range(num_matches), num_tre_points)
    other_idx = list(all_idx - set(tre_idx))
    moving_pts_for_reg, fixed_pts_for_reg = moving_matches[other_idx], fixed_matches[other_idx]

    tform = estimate_transform('similarity', src=moving_pts_for_reg, dst=fixed_pts_for_reg)
    aligned_moving = warp(moving, tform.inverse, output_shape=fixed.shape)

    return tform, aligned_moving, [moving_matches[tre_idx], fixed_matches[tre_idx]], [moving_pts_for_reg, fixed_pts_for_reg]



def register_DAPI_HnE(fixed, moving, adv_tform=None, feature_tform=None, intensity_tform=None, mpp=None, max_ratio=0.6, n_octaves=3, n_scales=5, scale_factor=4):
    """
    Main registration function to detect features, register DAPI stained multiplexed image to HnE stained image using initial feature based 
    registration followed by optional advanced feature based and/or intensity based registration.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - adv_tform (str, optional) : type of advanced registration to apply ('feature' or 'intensity')
    - feature_tform (str, optional) : type of transformation for advanced feature based registration
                                    ('affine', 'projective')
    - intensity_tform (str, optional) : type of transformation for intensity based registration
                                    ('rigid', 'affine', 'bspline',  'r-af-bs', 'af-bs', 'r-af', 'r-bs')
    - mpp (float, optional) : pixel size only required for intensity based registration
    - max_ratio (float) : maximum ratio for descriptor matching
    - n_octaves (int) : number of octaves for SIFT
    - n_scales (int) : number of scales per octave for SIFT
    - scale_factor (int) : factor to downscale images for feature extraction

    Returns:
    - transformation_maps (dict) : dictionary of transformation maps (skimage Transform objects or itk Transform objects)
                                    'initial similarity' : skimage Transform object for initial feature based registration
                                    'intensity based' : dictionary of intensity based registration transforms (if any)
                                                        'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
                                                        OR
                                    'affine' or 'projective' : skimage Transform object
    - registered_imgs (dict) : dictionary of registered images after each registration step
                                    'intensity based' : dictionary of intensity based registered images (if any)
                                                        'rigid' and/or 'affine' and/or 'bspline' : np.array of registered image
                                                        OR
                                    'affine' or 'projective' : np.array of registered image
    - tre_points (list of np.array) : points selected for TRE computation [moving_points, fixed_points]
    """

    transformations_map = {}
    registered_imgs = {}

    tform_map_init, moving_img_aligned, [moving_tre_pts, fixed_tre_pts], [moving_reg_pts, fixed_reg_pts] = register_init_feature_based(fixed, moving, max_ratio=max_ratio, 
                                                                                                                                       n_octaves=n_octaves, n_scales=n_scales, 
                                                                                                                                       scale_factor=scale_factor)
    transformations_map['initial similarity'] = tform_map_init
    registered_imgs['initial similarity'] = moving_img_aligned

    print('Initial feature based registration completed.')

    if adv_tform == 'feature':

        if feature_tform is None:
            raise ValueError("feature_tform must be provided for advanced feature based registration")

        tform_map_feat, moving_img_aligned = register_feature_based(fixed, moving, feature_tform, moving_reg_pts, fixed_reg_pts)
        transformations_map[feature_tform] = tform_map_feat
        registered_imgs[feature_tform] = moving_img_aligned

        print('Advanced feature based registration completed.')


    elif adv_tform == 'intensity':
        if mpp is None or intensity_tform is None:
            raise ValueError("mpp and intensity_tform must be provided for intensity based registration")

        tform_maps, reg_imgs = register_adv_intensity_based(
            fixed,
            moving_img_aligned,
            mpp,
            intensity_tform
        )

        transformations_map['intensity based'] = tform_maps
        registered_imgs['intensity based'] = reg_imgs

        print('Follow-up intensity based registration completed.')

    else:
        print('No advanced registration applied.')

    return transformations_map, registered_imgs, [moving_tre_pts, fixed_tre_pts]



    