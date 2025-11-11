from importlib import resources
import numpy as np
import random
from skimage.util import img_as_float32
from skimage.transform import resize, estimate_transform, warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage import measure
import itk



def apply_tform_adv(moving_img_itk, transform_map):
    
    #no_of_transforms = transform_map.GetNumberOfParameterMaps()

    transformed_img = {}
    result = moving_img_itk
    #for n in range(no_of_transforms):
    for key, value in transform_map.items():
        # single_transform = itk.ParameterObject.New()
        # single_transform.AddParameterMap(transform_map.GetParameterMap(n))
        result = itk.transformix_filter(result, value, log_to_console=False)
        transformed_img[key] = itk.GetArrayFromImage(result)

    return transformed_img



def register_adv_intensity_based(fixed, moving, mpp, transformation):
    
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
    else:
        raise ValueError("transformation for intensity based registration must be one of 'rigid', 'affine', 'bspline', 'r-af-bs', or 'af-bs'.")

    # global_trf_map = itk.ParameterObject.New()

    parameter_maps = []
    for tform in transform_scheme:
        with resources.path("registration.tform_params_adv", f"{tform}.txt") as p:
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

        #global_trf_map.AddParameterMap(result_trf_params.GetParameterMap(0))
        global_trf_map[tfname] = result_trf_params

    registered_imgs = apply_tform_adv(moving_itk, global_trf_map)

    return global_trf_map, registered_imgs



def features_with_SIFT(fixed, moving, max_ratio=0.6, n_octaves=3, n_scales=5):
    fixedX, fixedY = fixed.shape
    movingX, movingY = moving.shape
    scale_factor = 4

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
    tform = estimate_transform(transform_type, src=moving_matches, dst=fixed_matches)
    aligned_moving = warp(moving, tform.inverse, output_shape=fixed.shape)

    return tform, aligned_moving


def register_init_feature_based(fixed, moving):


    [moving_matches, fixed_matches] = features_with_SIFT(fixed, moving)

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



def register_DAPI_HnE(fixed, moving, adv_tform=None, feature_tform=None, intensity_tform=None, mpp=None):

    transformations_map = {}
    registered_imgs = {}

    tform_map_init, moving_img_aligned, [moving_tre_pts, fixed_tre_pts], [moving_reg_pts, fixed_reg_pts] = register_init_feature_based(fixed, moving)
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



    