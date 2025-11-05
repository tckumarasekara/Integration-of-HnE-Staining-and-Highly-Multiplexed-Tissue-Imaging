import glob
from pathlib import Path
from importlib import resources
import numpy as np
import tifffile as tifff
from skimage.util import img_as_float32
from skimage.transform import resize, estimate_transform, warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage import measure
import itk
import os



def apply_tform_adv(moving_img_itk, transform_map):
    
    no_of_transforms = transform_map.GetNumberOfParameterMaps()

    transformed_img = []
    result = moving_img_itk
    for n in range(no_of_transforms):
        single_transform = itk.ParameterObject.New()
        single_transform.AddParameterMap(transform_map.GetParameterMap(n))
        result = itk.transformix_filter(result, single_transform, log_to_console=False)
        transformed_img.append(itk.GetArrayFromImage(result))

    print('Follow-up intensity based registration completed.')

    return transformed_img



def register_adv_intensity_based(fixed, moving, mpp, transformation):
    
    if transformation == 'rigid':
        transform_scheme = ["01_Rigid"]
    elif transformation == 'affine':
        transform_scheme = ["01_Rigid", "02_Affine"]
    elif transformation == 'bspline':
        transform_scheme = ["01_Rigid", "02_Affine", "03_BSpline"]
    else:
        raise ValueError("transformation for intensity based registration must be one of 'rigid', 'affine', or 'bspline'")

    global_trf_map = itk.ParameterObject.New()

    parameter_maps = []
    for tform in transform_scheme:
        with resources.path("registration.tform_params_adv", f"{tform}.txt") as p:
            parameter_maps.append(str(p))

    fixed_itk = itk.GetImageFromArray(img_as_float32(fixed))
    fixed_itk.SetSpacing([mpp,mpp])

    moving_itk = itk.GetImageFromArray(img_as_float32(moving))
    moving_itk.SetSpacing([mpp,mpp])

    for Reg in parameter_maps:
        reg_map = itk.ParameterObject.New()
        reg_map.AddParameterFile(str(Reg))

        moving_itk, result_trf_params = itk.elastix_registration_method(
            fixed_itk, 
            moving_itk,
            parameter_object = reg_map,
            log_to_console=False
        )

        global_trf_map.AddParameterMap(result_trf_params.GetParameterMap(0))

    registered_imgs = apply_tform_adv(moving_itk, global_trf_map)

    return global_trf_map, registered_imgs



def features_with_SIFT(dapi_img, hne_img, max_ratio=0.6, n_octaves=3, n_scales=5):
    dapiX, dapiY = dapi_img.shape
    hneX, hneY = hne_img.shape
    scale_factor = 4

    # Resize the images to reduce memory usage
    dapi_scaled = resize(dapi_img, (dapiX // scale_factor, dapiY // scale_factor), anti_aliasing=True)
    hne_scaled = resize(hne_img, (hneX // scale_factor, hneY // scale_factor), anti_aliasing=True)

    descriptor_extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales)

    descriptor_extractor.detect_and_extract(hne_scaled)
    keypoints1, descriptors1 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(dapi_scaled)
    keypoints2, descriptors2 = descriptor_extractor.keypoints, descriptor_extractor.descriptors

    matches12 = match_descriptors(
        descriptors1, descriptors2, max_ratio=max_ratio, cross_check=True
    )

    # Extract matched keypoints
    src, dst = keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]

    dst, src = dst * scale_factor, src * scale_factor

    # Compute inliers using RANSAC 
    _, inliers = measure.ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)
    
    hnetemp_matches = src[inliers] 
    dapitemp_matches = dst[inliers] 
    
    hne_matches = hnetemp_matches[:, [1, 0]].copy()
    dapi_matches = dapitemp_matches[:, [1, 0]].copy()

    return [hne_matches, dapi_matches]



def register_init_feature_based(dapi_img, hne_img, tform_type):


    [hne_matches, dapi_matches] = features_with_SIFT(dapi_img, hne_img)

    tform = estimate_transform(tform_type, src=hne_matches, dst=dapi_matches)
    aligned_hne = warp(hne_img, tform.inverse, output_shape=dapi_img.shape)

    print('Initial feature based registration completed.')

    return tform, aligned_hne



def register_DAPI_HnE(dapi_img, hne_img, tform_init='similarity', tform_adv=None, mpp=None):

    transformations_map = []
    registered_imgs = []

    tform_map_init, hne_img_aligned = register_init_feature_based(dapi_img, hne_img, tform_init)
    transformations_map.append(tform_map_init)
    registered_imgs.append(hne_img_aligned)

    if tform_adv is not None:
        if mpp is None:
            raise ValueError("mpp must be provided for intensity based registration")
        tform_maps, reg_imgs = register_adv_intensity_based(
            dapi_img,
            hne_img_aligned,
            mpp,
            tform_adv
        )

        transformations_map.append(tform_maps)
        registered_imgs.extend(reg_imgs)

    return transformations_map, registered_imgs



    