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

    if matches12.shape[0] < 3:
        raise ValueError("Not enough matching points found between images for reliable registration.")

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



def register_feature_based(dapi_img, hne_img, transform_type, hne_matches, dapi_matches):
    tform = estimate_transform(transform_type, src=hne_matches, dst=dapi_matches)
    aligned_hne = warp(hne_img, tform.inverse, output_shape=dapi_img.shape)

    return tform, aligned_hne



def register_init_feature_based(dapi_img, hne_img):


    [hne_matches, dapi_matches] = features_with_SIFT(dapi_img, hne_img)

    num_matches = hne_matches.shape[0]
    num_tre_points = min(6, num_matches - 3, num_matches // 2)
    all_idx = set(range(num_matches))
    tre_idx = random.sample(range(num_matches), num_tre_points)
    other_idx = list(all_idx - set(tre_idx))
    hne_pts_for_reg, dapi_pts_for_reg = hne_matches[other_idx], dapi_matches[other_idx]

    tform = estimate_transform('similarity', src=hne_pts_for_reg, dst=dapi_pts_for_reg)
    aligned_hne = warp(hne_img, tform.inverse, output_shape=dapi_img.shape)

    return tform, aligned_hne, [hne_matches[tre_idx], dapi_matches[tre_idx]], [hne_pts_for_reg, dapi_pts_for_reg]



def register_DAPI_HnE(dapi_img, hne_img, adv_tform=None, feature_tform=None, intensity_tform=None, mpp=None):

    transformations_map = {}
    registered_imgs = {}

    tform_map_init, hne_img_aligned, [hne_tre_pts, dapi_tre_pts], [hne_reg_pts, dapi_reg_pts] = register_init_feature_based(dapi_img, hne_img)
    transformations_map['initial similarity'] = tform_map_init
    registered_imgs['initial similarity'] = hne_img_aligned

    print('Initial feature based registration completed.')

    if adv_tform == 'feature':

        if feature_tform is None:
            raise ValueError("feature_tform must be provided for advanced feature based registration")

        tform_map_feat, hne_img_aligned = register_feature_based(dapi_img, hne_img, feature_tform, hne_reg_pts, dapi_reg_pts)
        transformations_map[feature_tform] = tform_map_feat
        registered_imgs[feature_tform] = hne_img_aligned

        print('Advanced feature based registration completed.')


    elif adv_tform == 'intensity':
        if mpp is None or intensity_tform is None:
            raise ValueError("mpp and intensity_tform must be provided for intensity based registration")

        tform_maps, reg_imgs = register_adv_intensity_based(
            dapi_img,
            hne_img_aligned,
            mpp,
            intensity_tform
        )

        transformations_map['intensity based'] = tform_maps
        registered_imgs['intensity based'] = reg_imgs

        print('Follow-up intensity based registration completed.')

    else:
        print('No advanced registration applied.')

    return transformations_map, registered_imgs, [hne_tre_pts, dapi_tre_pts]



    