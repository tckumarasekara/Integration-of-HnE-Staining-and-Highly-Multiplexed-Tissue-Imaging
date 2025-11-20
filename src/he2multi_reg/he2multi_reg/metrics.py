import numpy as np
import itk
from skimage.util import img_as_float32
from scipy.stats import entropy

def compute_TRE(tform_maps, tre_points, fixed, mpp=None):

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