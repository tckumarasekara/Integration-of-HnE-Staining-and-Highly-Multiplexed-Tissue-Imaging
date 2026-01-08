import numpy as np
import itk
from skimage.util import img_as_float32
from scipy.stats import entropy

def compute_TRE(tform_maps, tre_points, fixed, mpp=None):
    """
    Compute Target Registration Error (TRE) before and after each registration.

    Parameters:
    - tform_maps (dict) : dictionary of transformation maps (skimage Transform objects or itk Transform objects)
                            'initial similarity' : skimage Transform object for initial feature based registration
                            'intensity based' : dictionary of intensity based registration transforms (if any)
                                                'rigid' and/or 'affine' and/or 'bspline' : itk Transform object
                                                OR
                            'affine' or 'projective' : skimage Transform object                                         
    - tre_points (tuple of arrays) : (source points, destination points) for TRE computation
    - fixed (np.array) : fixed image
    - mpp (float, optional) : pixel size only required for TRE computation after intensity based registration

    Returns:
    - tre (dict) : dictionary of TRE values before and after each registration step
                            'before registration' : float rTRE before registration
                            'initial similarity' : float rTRE after initial feature based registration
                            'rigid' and/or 'affine' and/or 'bspline' OR 'affine' or 'projective' : float rTRE after 
                                                                                                    each registration (if any)                                                                                               
    """

    src_points, dst_points = tre_points
    tre = {}
    h, w = fixed.shape   
    diagonal = np.sqrt(h**2 + w**2)

    if len(src_points) != len(dst_points):
        raise ValueError("Same number of source and destination points must be provided.")
    
    if len(src_points) < 3:
        raise ValueError("At least three points are required to compute TRE.")
    
    tre_temp = np.mean(np.linalg.norm(np.array(src_points) - np.array(dst_points), axis=1))
    tre_temp /= diagonal    
    tre['before registration'] = tre_temp
    print("rTRE before registration: ", tre_temp)
    
    tform = tform_maps['initial similarity']
    transformed_src = np.array(tform(src_points), dtype=float)
    tre_temp = np.mean(np.linalg.norm(transformed_src - dst_points, axis=1))
    tre_temp /= diagonal
    tre['initial similarity'] = tre_temp
    print("rTRE after initial feature based registration: ", tre_temp)

    if 'intensity based' not in tform_maps and len(tform_maps) > 1:
        key = list(tform_maps.keys())[1]
        tform = tform_maps[key]
        transformed_src = np.array(tform(src_points), dtype=float)
        tre_temp = np.mean(np.linalg.norm(transformed_src - dst_points, axis=1))
        tre_temp /= diagonal
        tre[key] = tre_temp
        print("rTRE after feature based", key, "transformation: ", tre_temp)

    elif 'intensity based' in tform_maps:

        if mpp is None:
            raise ValueError("mpp must be provided for TRE computation after intensity based registration")
        
        img_shape = fixed.shape
        
        for tform_name, tform in tform_maps['intensity based'].items():

            coord_list = []
            removed_pts_idx = []

            for idx, (pt1, pt2) in enumerate(transformed_src):

                if np.isnan(pt1) or np.isnan(pt2):
                    print("Skipping NaN point:", pt1, pt2)
                    continue

                x, y = int(round(pt1)), int(round(pt2))

                if not (0 <= x < img_shape[1] and 0 <= y < img_shape[0]):
                    print("Skipping out-of-bounds point:", x, y)
                    continue

                img = np.zeros(img_shape, dtype=np.float32)
                img[y, x] = 1.0
                moving_mask = itk.GetImageFromArray(img_as_float32(img))
                moving_mask.SetSpacing([mpp,mpp])

                tform.SetParameter("FinalBSplineInterpolationOrder", '0')

                result_moving_mask = itk.transformix_filter(
                    moving_mask, 
                    tform,
                    log_to_console=False
                )

                arr = itk.GetArrayFromImage(result_moving_mask)
                ys, xs = np.where(arr > 0.9)

                if len(xs) == 0 or len(ys) == 0:
                    print("No transformed point found for original point:", x, y)
                    removed_pts_idx.append(idx)
                    continue

                x_new, y_new = np.mean(xs), np.mean(ys) # if multiple pixels
                coord_list.append([x_new, y_new])

            transformed_src = np.array(coord_list)
            removed_pts_idx_arr = np.array(removed_pts_idx, dtype=int)
            try:
                dst_points = np.delete(dst_points, removed_pts_idx_arr, axis=0)
                tre_temp = np.mean(np.linalg.norm(transformed_src - dst_points, axis=1))
                tre_temp /= diagonal
                print(f"rTRE after follow-up intensity based {tform_name} transformation: ", tre_temp)
                tre[tform_name] = tre_temp
            except:
                print("rTRE not computed for intensity based registration due to unexpected error.")

    return tre



def mutual_information_metric(fixed, moving, bins):
    """
    Compute the normalized Mutual Information (MI) between two images.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - bins (int) : number of bins for histogram computation

    Returns:
    - n_mutual_info (float) : normalized Mutual Information value
    """

    fy, fx = fixed.shape
    my, mx = moving.shape

    if fy != my or fx != mx:
        min_y = min(fy, my)
        min_x = min(fx, mx)

        fixed = fixed[:min_y, :min_x]
        moving = moving[:min_y, :min_x]

    # Compute joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(fixed.ravel(), moving.ravel(), bins=bins)
    
    # Normalize to get joint probabilities
    pxy = hist_2d / np.sum(hist_2d)
    
    # Marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropies
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = entropy(pxy.ravel())
    
    # Mutual Information
    mutual_info = Hx + Hy - Hxy
    n_mutual_info = mutual_info / np.mean([Hx, Hy]) 
    
    return n_mutual_info



def compute_mutual_information(fixed, moving, transformed_imgs, bins = 50):
    """
    Compute normalized Mutual Information (MI) before and after each registration.

    Parameters:
    - fixed (np.array) : fixed image
    - moving (np.array) : moving image
    - transformed_imgs (dict) : dictionary of transformed images after each registration step
                                'initial similarity' : np.array of image after initial feature based registration
                                'intensity based' : dictionary of intensity based registered images (if any)
                                                    'rigid' and/or 'affine' and/or 'bspline' : np.array of image
                                                OR
                                'affine' or 'projective' : np.array of image after feature based registration
    - bins (int, optional) : number of bins for histogram computation (default is 50)

    Returns:
    - mi_scores (dict) : dictionary of normalized MI values before and after each registration step
                                'before registration' : float normalized MI before registration
                                'initial similarity' : float normalized MI after initial feature based registration
                                'rigid' and/or 'affine' and/or 'bspline' OR 'affine' or 'projective' : float normalized MI after 
                                                                                                        each registration (if any)                                                                                                 
    """

    mi_scores = {}
    
    mi_before = mutual_information_metric(fixed, moving, bins)
    mi_scores['before registration'] = mi_before
    print("normalized MI before registration: ", mi_before)

    tform_img = transformed_imgs['initial similarity']
    mi_after_init = mutual_information_metric(fixed, tform_img, bins)
    mi_scores['initial similarity'] = mi_after_init
    print("normalized MI after initial feature based registration: ", mi_after_init)

    if 'intensity based' not in transformed_imgs and len(transformed_imgs) > 1:
        key = list(transformed_imgs.keys())[1]
        tform_img = transformed_imgs[key]
        mi_after_feat = mutual_information_metric(fixed, tform_img, bins)
        mi_scores[key] = mi_after_feat
        print("normalized MI after feature based", key, "transformation: ", mi_after_feat)

    elif 'intensity based' in transformed_imgs:

        for tform_name, tform_img in transformed_imgs['intensity based'].items():
            mi_after_intensity = mutual_information_metric(fixed, tform_img, bins)
            mi_scores[tform_name] = mi_after_intensity
            print("normalized MI after follow-up intensity based", tform_name, "transformation: ", mi_after_intensity)

    return mi_scores