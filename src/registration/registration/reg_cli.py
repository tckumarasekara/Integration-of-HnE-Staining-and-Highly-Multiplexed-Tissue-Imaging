import argparse
import os
import json
from tifffile import imwrite
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
    parser.add_argument("--intensity_tform", type=str, choices=['rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs', 'r-bs', 'r-af'], default=None, help="Intensity transformation method for follow-up intensity based registration")

    args = parser.parse_args()

    transformation_maps, registered_imgs, final_img, tre, mi = registration_pipeline(
        args.fixed_path,
        args.moving_path,
        args.fixed_px_sz,
        args.moving_px_sz,
        args.fixed_img,
        adv_tform=args.adv_tform,
        feature_tform=args.feature_tform,
        intensity_tform=args.intensity_tform
    )

    output_folder_path = os.path.join(args.output_folder, "results")
    os.makedirs(output_folder_path, exist_ok=True)

    # save registration metrics
    metrics_output_path = os.path.join(output_folder_path, "registration_metrics.json")
    
    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered images
    img_output_folder = os.path.join(output_folder_path, "registered_images")
    os.makedirs(img_output_folder, exist_ok=True)

    init_reg_img = registered_imgs['initial similarity']
    init_reg_img_path = os.path.join(img_output_folder, "initial_feature_based_similarity_registered_image.tif")
    imwrite(init_reg_img_path, init_reg_img)

    if args.adv_tform == 'intensity':
        for tform_name, img in registered_imgs['intensity based'].items():
            img_path = os.path.join(img_output_folder, f"intensity_based_{tform_name}_registered_image.tif")
            imwrite(img_path, img)

    elif args.adv_tform == 'feature':
        key = list(transformation_maps.keys())[1]
        adv_reg_img = registered_imgs[key]
        adv_reg_img_path = os.path.join(img_output_folder, f"feature_based_{key}_registered_image.tif")
        imwrite(adv_reg_img_path, adv_reg_img)

    final_img_path = os.path.join(img_output_folder, "0_final_channel_image.tif")
    imwrite(final_img_path, final_img)

    print(f"Registered images saved to {img_output_folder}")

    # save transformation maps
    tform_output_folder = os.path.join(output_folder_path, "transformation_maps")
    os.makedirs(tform_output_folder, exist_ok=True)

    init_tform_map = transformation_maps['initial similarity']
    init_tform_map_path = os.path.join(tform_output_folder, "initial_feature_based_similarity_transformation_map.txt")
    with open(init_tform_map_path, "w") as f:
        f.write(str(init_tform_map.params))
    
    if args.adv_tform == 'intensity':
        for tform_name, tform_map in transformation_maps['intensity based'].items():
            tform_map_path = os.path.join(tform_output_folder, f"intensity_based_{tform_name}_transformation_map.txt")
            with open(tform_map_path, "w") as f:
                for item in tform_map.GetParameterMap(0).items():
                    f.write(f"{item}\n")

    elif args.adv_tform == 'feature':
        key = list(transformation_maps.keys())[1]
        adv_tform_map = transformation_maps[key]
        adv_tform_map_path = os.path.join(tform_output_folder, f"feature_based_{key}_transformation_map.txt")
        with open(adv_tform_map_path, "w") as f:
            f.write(str(adv_tform_map.params))

    print(f"Transformation maps saved to {tform_output_folder}")


if __name__ == "__main__":
    main()