import sys
sys.path.insert(0, "/home-link/zxovq55/Masters-Thesis/src/registration")

import numpy as np
import os
import pickle

import importlib, registration 
importlib.reload(registration)
importlib.reload(registration.reg)
importlib.reload(registration.metrics)
importlib.reload(registration.regPipeline)
importlib.reload(registration.preprocess)

fixed_px_sz = 0.209877
moving_px_sz = 0.5023

num_imgs = 3
num_runs = 2

results_dir = os.path.join("result_plots_metrics")
os.makedirs(results_dir, exist_ok=True)

octaves = [2, 3, 4]
scales = [2, 3, 4, 5]
resolutions = [2, 4, 8, 12, 16]

tre_dict = {}
mi_dict ={}

def add_value(d, key, value):
    value = np.array([value])        
    if key in d:
        d[key] = np.concatenate([d[key], value])
    else:
        d[key] = value

for img_idx in range(num_imgs):

    fixed_init, moving_init = registration.load_and_scale_images(f"../../data/dapi_{img_idx+1}.tif", 
                                                                                 f"../../data/hne_{img_idx+1}.tif", 
                                                                                fixed_px_sz, 
                                                                                moving_px_sz)
        
    fixed_prepr = fixed_init
    moving_prepr = registration.colour_deconvolusion_preprocessing_HnE(moving_init)
    
    for run_idx in range(num_runs):
        for octave in octaves:
            for scale in scales:
                for resolution in resolutions:

                    print(f"Processing img_{img_idx+1}, run_{run_idx+1}, octave_{octave}, scale_{scale}, resolution_{resolution}")

                    try:

                        transformation_maps, registered_imgs, tre_pts = registration.register_DAPI_HnE(fixed_prepr, moving_prepr, max_ratio=0.6, n_octaves=octave, n_scales=scale, scale_factor=resolution) 

                        try:
                            tre = registration.compute_TRE(transformation_maps, tre_pts, fixed_prepr, mpp=moving_px_sz)
                        except ValueError as e:
                            print("TRE computation skipped:", e)
                            tre = None  
                        except Exception as e:
                            print("An unexpected error occurred during TRE computation:", e)
                            tre = None

                        try:
                            mi = registration.compute_mutual_information(fixed_prepr, moving_prepr, registered_imgs)
                        except Exception as e:
                            print("An unexpected error occurred during mutual information computation:", e)
                            mi = None

                        add_value(tre_dict, (octave, scale, resolution), tre['initial similarity'])
                        add_value(mi_dict, (octave, scale, resolution), mi['initial similarity'])

                    except:
                        add_value(tre_dict, (octave, scale, resolution), None)
                        add_value(mi_dict, (octave, scale, resolution), None)


with open(os.path.join(results_dir, "tre_dict.pkl"), "wb") as f:
    pickle.dump(tre_dict, f)
with open(os.path.join(results_dir, "mi_dict.pkl"), "wb") as f:
    pickle.dump(mi_dict, f)
