import sys
sys.path.insert(0, "/home-link/zxovq55/Masters-Thesis/src/he2multi_reg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tifffile import imread
from skimage.transform import resize, warp
import numpy as np
import os
from skimage.util import img_as_float32
import itk
import time
import pickle

import importlib, he2multi_reg 
importlib.reload(he2multi_reg)
importlib.reload(he2multi_reg.reg)
importlib.reload(he2multi_reg.metrics)
importlib.reload(he2multi_reg.regPipeline)
importlib.reload(he2multi_reg.preprocess)

num_runs = 5
num_imgs = 3

fixed_px_sz = 0.209877
moving_px_sz = 0.5023
adv_tform = 'intensity'
intensity_tform = 'r-af-bs'

tre_bg_dict = {}
tre_ftr_dict = {}
tre_r_dict = {}
tre_af_dict = {}
tre_bs_dict = {}
mi_bg_dict = {}
mi_ftr_dict = {}
mi_r_dict = {}
mi_af_dict = {}
mi_bs_dict = {}
reg_time_dict = {}
reg2_time_dict ={}
reg_metric_time_dict = {}

results_dir = os.path.join("result_plots_metrics")

os.makedirs(results_dir, exist_ok=True)

for img_idx in range(num_imgs):
    tre_bg_list = []
    tre_ftr_list = []
    tre_r_list = []
    tre_af_list = []
    tre_bs_list = []
    mi_bg_list = []
    mi_ftr_list = []
    mi_r_list = []
    mi_af_list = []
    mi_bs_list = []
    reg_time_list = []
    reg2_time_list = []
    reg_metric_time_list = []

    for run_idx in range(num_runs):

        start_time = time.time()

        output_folder = os.path.join(results_dir, f"img_{img_idx}_run_{run_idx}")
        os.makedirs(output_folder, exist_ok=True)

        fixed_init, moving_init = he2multi_reg.load_and_scale_images(f"../../data/dapi_{img_idx+1}.tif", 
                                                                     f"../../data/hne_{(img_idx+1)}.tif", 
                                                                     fixed_px_sz, 
                                                                     moving_px_sz)
        
        fixed_prepr = fixed_init
        moving_prepr = he2multi_reg.colour_deconvolusion_preprocessing_HnE(moving_init)

        if len(fixed_init.shape) == 2:
            h, w = fixed_init.shape
        else:
            h, w, c = fixed_init.shape

        transformation_maps, registered_imgs, tre_pts = he2multi_reg.register_DAPI_HnE(fixed_prepr, 
                                                                                       moving_prepr, 
                                                                                       adv_tform, 
                                                                                       None, 
                                                                                       intensity_tform, 
                                                                                       mpp=moving_px_sz)
        
        moved_init = warp(moving_init, transformation_maps['initial similarity'].inverse, output_shape=(h, w, moving_init.shape[2]) if len(moving_init.shape) == 3 else (h, w))

        mid_time = time.time()

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


        mid_time2 = time.time()

        try:
            tre = he2multi_reg.compute_TRE(transformation_maps, tre_pts, fixed_prepr, mpp=moving_px_sz)
        except ValueError as e:
            print("TRE computation skipped:", e)
            tre = None  
        except Exception as e:
            print("An unexpected error occurred during TRE computation:", e)
            tre = None

        try:
            mi = he2multi_reg.compute_mutual_information(fixed_prepr, moving_prepr, registered_imgs)
        except Exception as e:
            print("An unexpected error occurred during mutual information computation:", e)
            mi = None

        end_time = time.time()

        reg_time = mid_time - start_time
        reg2_time = mid_time2 - mid_time
        reg_metric_time = end_time - mid_time2

        reg_time_list.append(reg_time)
        reg2_time_list.append(reg2_time)
        reg_metric_time_list.append(reg_metric_time)
        tre_bg_list.append(tre['before he2multi_reg'])
        tre_ftr_list.append(tre['initial similarity'])
        tre_r_list.append(tre['rigid'])
        tre_af_list.append(tre['affine'])
        tre_bs_list.append(tre['bspline'])
        mi_bg_list.append(mi['before he2multi_reg'])
        mi_ftr_list.append(mi['initial similarity'])
        mi_r_list.append(mi['rigid'])
        mi_af_list.append(mi['affine'])
        mi_bs_list.append(mi['bspline'])
        print(f"Completed he2multi_reg for img_{img_idx+1} run_{run_idx+1}")

    tre_bg_dict[f"img_{img_idx+1}"] = np.array(tre_bg_list)
    tre_ftr_dict[f"img_{img_idx+1}"] = np.array(tre_ftr_list)
    tre_r_dict[f"img_{img_idx+1}"] = np.array(tre_r_list)
    tre_af_dict[f"img_{img_idx+1}"] = np.array(tre_af_list)
    tre_bs_dict[f"img_{img_idx+1}"] = np.array(tre_bs_list)
    mi_bg_dict[f"img_{img_idx+1}"] = np.array(mi_bg_list)
    mi_ftr_dict[f"img_{img_idx+1}"] = np.array(mi_ftr_list)
    mi_r_dict[f"img_{img_idx+1}"] = np.array(mi_r_list)
    mi_af_dict[f"img_{img_idx+1}"] = np.array(mi_af_list)
    mi_bs_dict[f"img_{img_idx+1}"] = np.array(mi_bs_list)

    reg_time_dict[f"img_{img_idx+1}"] = np.array(reg_time_list)
    reg2_time_dict[f"img_{img_idx+1}"] = np.array(reg2_time_list)
    reg_metric_time_dict[f"img_{img_idx+1}"] = np.array(reg_metric_time_list)

    with open(os.path.join(results_dir, f"{img_idx}_tre_bg_dict.pkl"), "wb") as f:
        pickle.dump(tre_bg_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_tre_ftr_dict.pkl"), "wb") as f:
        pickle.dump(tre_ftr_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_tre_r_dict.pkl"), "wb") as f:
        pickle.dump(tre_r_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_tre_af_dict.pkl"), "wb") as f:
        pickle.dump(tre_af_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_tre_bs_dict.pkl"), "wb") as f:
        pickle.dump(tre_bs_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_mi_bg_dict.pkl"), "wb") as f:
        pickle.dump(mi_bg_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_mi_ftr_dict.pkl"), "wb") as f:
        pickle.dump(mi_ftr_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_mi_r_dict.pkl"), "wb") as f:
        pickle.dump(mi_r_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_mi_af_dict.pkl"), "wb") as f:
        pickle.dump(mi_af_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_mi_bs_dict.pkl"), "wb") as f:
        pickle.dump(mi_bs_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_reg_time_dict.pkl"), "wb") as f:
        pickle.dump(reg_time_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_reg2_time_dict.pkl"), "wb") as f:
        pickle.dump(reg2_time_dict, f)
    with open(os.path.join(results_dir, f"{img_idx}_reg_metric_time_dict.pkl"), "wb") as f:
        pickle.dump(reg_metric_time_dict, f)