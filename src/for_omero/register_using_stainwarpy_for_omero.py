import os
import glob
import shutil
import subprocess

HNE_DIR = "hne_init"
MULTIPLEX_BASE = "images_2751_2790"
RESULTS_BASE = "results"

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

    print(f"\n[RUN] {hne_name}")

    cmd = [
        "stainwarpy", "register",
        multiplex_path,
        hne_path,
        multiplex_dir,
        "multiplexed",
        "multiplexed",
        "--hne-px-sz", HNE_PX_SZ
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] stainwarpy failed for {hne_name}")
        print(e.stderr)
        continue

    # rename registered image 
    registered_candidates = glob.glob(
        os.path.join(multiplex_dir, "0_final_channel_image.ome.tif")
    )

    try:
        if registered_candidates:
            old_path = registered_candidates[0]
            new_name = f"{after_id}.ome.tif"
            new_path = os.path.join(multiplex_dir, new_name)

            os.rename(old_path, new_path)
    except:
        print(f"[WARN] No registered image found for {hne_name}")

    # move transformation + metrics 
    result_dir = os.path.join(RESULTS_BASE, f"Image_{sample_id}")
    os.makedirs(result_dir, exist_ok=True)

    try:
        for f in glob.glob(os.path.join(multiplex_dir, "*")):
            fname = os.path.basename(f).lower()
            if "registration_metrics_tform_map.json" in fname or "feature_based_transformation_map.npy" in fname:
                shutil.move(f, os.path.join(result_dir, os.path.basename(f)))
    except:
        print(f"[WARN] Failed to move transformation or metrics for {hne_name}")