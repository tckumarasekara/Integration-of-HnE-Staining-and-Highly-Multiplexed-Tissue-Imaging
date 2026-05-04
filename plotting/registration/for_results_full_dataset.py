import os
import glob
import shutil
import subprocess

HNE_DIR = "../../hne_init"
MULTIPLEX_BASE = "../../images_2751_2790"
RESULTS_BASE = "../../results_plots_full_dataset"
NUM_RUNS = 5

HNE_PX_SZ = "0.5023"

os.makedirs(RESULTS_BASE, exist_ok=True)

hne_images = glob.glob(os.path.join(HNE_DIR, "*.tif"))

for hne_path in hne_images:
    hne_name = os.path.basename(hne_path)

    try:
        sample_id = hne_name.split("_")[0]   
        after_id = hne_name.split(f"{sample_id}_", 1)[1].replace(".tif", "")
        roi_part = after_id.split("-", 1)[1]
    except Exception:
        print(f"[SKIP] Bad HNE filename: {hne_name}")
        continue

    multiplex_dir = os.path.join(MULTIPLEX_BASE, f"Image_{sample_id}")

    if not os.path.isdir(multiplex_dir):
        print(f"[SKIP] Missing multiplex folder: {multiplex_dir}")
        continue

    multiplex_candidates = glob.glob(os.path.join(multiplex_dir, "*.tif"))

    if not multiplex_candidates:
        print(f"[SKIP] No multiplex image for {hne_name}")
        continue

    multiplex_path = multiplex_candidates[0]

    for run_idx in range(NUM_RUNS):

        print(f"\n[RUN] {hne_name} num_run={run_idx+1}/{NUM_RUNS}")

        results_path = os.path.join(RESULTS_BASE, f"Image_{sample_id}_run_{run_idx+1}")
        os.makedirs(results_path, exist_ok=True)

        cmd = [
            "he2multi-reg", "register",
            multiplex_path,
            hne_path,
            results_path,
            "multiplexed",
            "--moving-px-sz", HNE_PX_SZ,
            "--adv-tform", "intensity",
            "--intensity-tform", "r-af-bs"
        ]

        try:
            subprocess.run(cmd, check=True)
            img_folder_path = os.path.join(results_path, "results/registered_images")
            shutil.rmtree(img_folder_path, ignore_errors=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] stainwarpy failed for {hne_name}")
            shutil.rmtree(results_path, ignore_errors=True)
            continue