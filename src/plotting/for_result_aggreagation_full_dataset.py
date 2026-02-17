import json
import os
import numpy as np

RESULTS_BASE = "../../results_plots_full_dataset"


tre = {
    "before registration": [],
    "initial similarity": [],
    "rigid": [],
    "affine": [],
    "bspline": []
}

mi = {
    "before registration": [],
    "initial similarity": [],
    "rigid": [],
    "affine": [],
    "bspline": []   
}

tre_sep = {
    "before registration": [],
    "initial similarity": [],
    "rigid": [],
    "affine": [],
    "bspline": []
}

mi_sep = {
    "before registration": [],
    "initial similarity": [],
    "rigid": [],
    "affine": [],
    "bspline": []   
}


for img_id in range(2751, 2791):
    img_rep_tre_bfr ,img_rep_tre_fr, img_rep_tre_r, img_rep_tre_af, img_rep_tre_bs = [], [], [], [], []
    img_reg_mi_bfr, img_reg_mi_fr, img_reg_mi_r, img_reg_mi_af, img_reg_mi_bs = [], [], [], [], []

    for run_idx in range(1, 6):
        results_path = os.path.join(RESULTS_BASE, f"Image_{img_id}_run_{run_idx}/results/registration_metrics.json")

        try:

            with open(results_path, "r") as f:
                results = json.load(f)
                tre["before registration"].append(results["TRE"]["before registration"])
                tre["initial similarity"].append(results["TRE"]["initial similarity"])
                tre["rigid"].append(results["TRE"]["rigid"])
                tre["affine"].append(results["TRE"]["affine"])
                tre["bspline"].append(results["TRE"]["bspline"])

                img_rep_tre_bfr.append(results["TRE"]["before registration"])
                img_rep_tre_fr.append(results["TRE"]["initial similarity"])
                img_rep_tre_r.append(results["TRE"]["rigid"])
                img_rep_tre_af.append(results["TRE"]["affine"])
                img_rep_tre_bs.append(results["TRE"]["bspline"])

                mi["before registration"].append(results["Mutual Information"]["before registration"])
                mi["initial similarity"].append(results["Mutual Information"]["initial similarity"])
                mi["rigid"].append(results["Mutual Information"]["rigid"])
                mi["affine"].append(results["Mutual Information"]["affine"])
                mi["bspline"].append(results["Mutual Information"]["bspline"])
    
                img_reg_mi_bfr.append(results["Mutual Information"]["before registration"])
                img_reg_mi_fr.append(results["Mutual Information"]["initial similarity"])
                img_reg_mi_r.append(results["Mutual Information"]["rigid"])
                img_reg_mi_af.append(results["Mutual Information"]["affine"])
                img_reg_mi_bs.append(results["Mutual Information"]["bspline"])
    
        except Exception as e:
            print(f"\n[READ] {img_id}_run_{run_idx}")
            print(f"[ERROR] Failed to read {results_path}: {e}")
            continue

    try:
        tre_sep["before registration"].append(np.mean(img_rep_tre_bfr)) if np.mean(img_rep_tre_bfr) is not None or np.mean(img_rep_tre_bfr) is not np.nan else None
        tre_sep["initial similarity"].append(np.mean(img_rep_tre_fr)) if np.mean(img_rep_tre_fr) is not None or np.mean(img_rep_tre_fr) is not np.nan else None
        tre_sep["rigid"].append(np.mean(img_rep_tre_r)) if np.mean(img_rep_tre_r) is not None or np.mean(img_rep_tre_r) is not np.nan else None
        tre_sep["affine"].append(np.mean(img_rep_tre_af)) if np.mean(img_rep_tre_af) is not None or np.mean(img_rep_tre_af) is not np.nan else None
        tre_sep["bspline"].append(np.mean(img_rep_tre_bs)) if np.mean(img_rep_tre_bs) is not None or np.mean(img_rep_tre_bs) is not np.nan else None

        mi_sep["before registration"].append(np.mean(img_reg_mi_bfr)) if np.mean(img_reg_mi_bfr) is not None or np.mean(img_reg_mi_bfr) is not np.nan else None
        mi_sep["initial similarity"].append(np.mean(img_reg_mi_fr)) if np.mean(img_reg_mi_fr) is not None or np.mean(img_reg_mi_fr) is not np.nan else None
        mi_sep["rigid"].append(np.mean(img_reg_mi_r)) if np.mean(img_reg_mi_r) is not None or np.mean(img_reg_mi_r) is not np.nan else None
        mi_sep["affine"].append(np.mean(img_reg_mi_af)) if np.mean(img_reg_mi_af) is not None or np.mean(img_reg_mi_af) is not np.nan else None
        mi_sep["bspline"].append(np.mean(img_reg_mi_bs)) if np.mean(img_reg_mi_bs) is not None or np.mean(img_reg_mi_bs) is not np.nan else None
        

    except:
        continue
        

with open("result_plots_metrics_full_dataset/tre.npy", "wb") as f:
    np.save(f, tre)

with open("result_plots_metrics_full_dataset/mi.npy", "wb") as f:
    np.save(f, mi)

with open("result_plots_metrics_full_dataset/tre_sep.npy", "wb") as f:
    np.save(f, tre_sep)

with open("result_plots_metrics_full_dataset/mi_sep.npy", "wb") as f:
    np.save(f, mi_sep)


#----------------------------------------------------------
# 2774, 2783 : no runs ------------------------------------
# 2772 : no runs included due to None type metrics --------
# 2782 : only 3 runs included due to None type metrics ----
#---------------------------------------------------------- 
            