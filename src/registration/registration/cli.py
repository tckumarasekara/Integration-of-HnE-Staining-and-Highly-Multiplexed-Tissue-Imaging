import argparse
import os
import json
from .regPipeline import registration_pipeline

def main():
    parser = argparse.ArgumentParser(description="Image Registration Pipeline")
    parser.add_argument("fixed_path", type=str, help="Path to the fixed image")
    parser.add_argument("moving_path", type=str, help="Path to the moving image")
    parser.add_argument("output_folder", type=str, help="Folder to save the registered images and metrics")
    parser.add_argument("fixed_px_sz", type=float, help="Pixel size of the fixed image")
    parser.add_argument("moving_px_sz", type=float, help="Pixel size of the moving image")
    parser.add_argument("fixed_img", type=str, choices=['multiplexed', 'hne'], help="Type of fixed image: 'multiplexed' or 'hne'")
    parser.add_argument("--adv_tform", type=str, choices=['feature', 'intensity'], default=None, help="Type of advanced transformation to apply")
    parser.add_argument("--feature_tform", type=str, choices=['affine', 'projective'], default=None, help="Feature transformation method for advanced feature based registration")
    parser.add_argument("--intensity_tform", type=str, choices=['rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs'], default=None, help="Intensity transformation method for follow-up intensity based registration")

    args = parser.parse_args()

    transformation_maps, registered_imgs, tre, mi = registration_pipeline(
        args.fixed_path,
        args.moving_path,
        args.fixed_px_sz,
        args.moving_px_sz,
        args.fixed_img,
        adv_tform=args.adv_tform,
        feature_tform=args.feature_tform,
        intensity_tform=args.intensity_tform
    )

    os.makedirs(args.output_folder, exist_ok=True)

    # save registration metrics
    metrics_output_path = os.path.join(args.output_folder, "registration_metrics.json")
    
    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered images
    