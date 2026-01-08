import numpy as np
import typer
import os
import json
from tifffile import imwrite, TiffFile
from skimage.transform import AffineTransform, ProjectiveTransform, resize
from .regPipeline import registration_pipeline
from .preprocess import extract_channel, load_image_data, save_ome_tiff, get_pixel_size_ome_tiff
from .reg import transform_seg_mask


app = typer.Typer(help="Register H&E stained images to multiplexed images using a feature and/or intensity based registration pipeline.")


@app.command(name="register")
def register(
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    moving_path: str = typer.Argument(..., help="Path to the moving image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder: str = typer.Argument(..., help="Folder to save the registered images and metrics"),
    fixed_img: str = typer.Argument(..., help="Type of fixed image: ['multiplexed', 'hne']"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)"),
    moving_px_sz: float = typer.Option(None, help="Pixel size of the moving image (if image is not .ome.tif)"),
    adv_tform: str = typer.Option(None, help="Type of advanced transformation to apply: ['feature', 'intensity']"),
    feature_tform: str = typer.Option(None, help="Feature transformation method for advanced feature based registration: ['affine', 'projective']"),
    intensity_tform: str = typer.Option(None, help="Intensity transformation method for follow-up intensity based registration: ['rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs', 'r-bs', 'r-af']"),
    intermediate_imgs: bool = typer.Option(False, help="Whether to save intermediate registered images: --intermediate-imgs or --no-intermediate-imgs", show_default=True),
):
    """
    Register H&E stained images to multiplexed images using a feature and/or intensity based registration pipeline. And save the registered images, transformation maps, and registration metrics to 
    the specified output folder.

    Parameters:
    - fixed_path (str): Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - moving_path (str): Path to the moving image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder (str): Folder to save the registered images and metrics
    - fixed_img (str): Type of fixed image: ['multiplexed', 'hne']
    - fixed_px_sz (float, optional): Pixel size of the fixed image (if image is not .ome.tif)
    - moving_px_sz (float, optional): Pixel size of the moving image (if image is not .ome.tif)
    - adv_tform (str, optional): Type of advanced transformation to apply: ['feature', 'intensity']
    - feature_tform (str, optional): Feature transformation method for advanced feature based registration: ['affine', 'projective']
    - intensity_tform (str, optional): Intensity transformation method for follow-up intensity based registration: ['rigid', 'affine', 'bspline', 'r-af-bs', 'af-bs', 'r-bs', 'r-af']
    - intermediate_imgs (bool, optional): Whether to save intermediate registered images (default is False)

    Returns:
     None. Saves registered images, transformation maps, and registration metrics to the specified output folder.
    """

    # run the pipeline
    transformation_maps, registered_imgs, final_img, tre, mi = registration_pipeline(
        fixed_path,
        moving_path,
        fixed_px_sz,
        moving_px_sz,
        fixed_img,
        adv_tform=adv_tform,
        feature_tform=feature_tform,
        intensity_tform=intensity_tform
    )

    output_folder_path = os.path.join(output_folder, "results")
    os.makedirs(output_folder_path, exist_ok=True)

    # save registration metrics
    metrics_output_path = os.path.join(output_folder_path, "registration_metrics.json")

    with open(metrics_output_path, "w") as f:
        json.dump({"TRE": tre, "Mutual Information": mi}, f)
    print(f"Registration metrics saved to {metrics_output_path}")

    # save registered images
    img_output_folder = os.path.join(output_folder_path, "registered_images")
    os.makedirs(img_output_folder, exist_ok=True)

    if intermediate_imgs:
        init_reg_img = registered_imgs['initial similarity']
        init_reg_img_path = os.path.join(img_output_folder, "initial_feature_based_similarity_registered_image.tif")
        imwrite(init_reg_img_path, init_reg_img)

        if adv_tform == 'intensity':
            for tform_name, img in registered_imgs['intensity based'].items():
                img_path = os.path.join(img_output_folder, f"intensity_based_{tform_name}_registered_image.tif")
                imwrite(img_path, img)

        elif adv_tform == 'feature':
            key = list(transformation_maps.keys())[1]
            adv_reg_img = registered_imgs[key]
            adv_reg_img_path = os.path.join(img_output_folder, f"feature_based_{key}_registered_image.tif")
            imwrite(adv_reg_img_path, adv_reg_img)

    ome_xml = None
    try:
        with TiffFile(moving_path) as ref:
            ome_xml = ref.ome_metadata
    except:
        pass

    final_img_path = os.path.join(img_output_folder, "0_final_channel_image.ome.tif")
    save_ome_tiff(final_img, final_img_path, physical_size_x=moving_px_sz, physical_size_y=moving_px_sz, source_ome_xml=ome_xml)

    print(f"Registered image/s saved to {img_output_folder}")

    # save transformation maps
    tform_output_folder = os.path.join(output_folder_path, "transformation_maps")
    os.makedirs(tform_output_folder, exist_ok=True)

    init_tform_map = transformation_maps['initial similarity']
    np.save(os.path.join(tform_output_folder, f"{1}_initial_feature_based_similarity_transformation_map.npy"), init_tform_map.params)

    if adv_tform == 'intensity':
        for idx, (tform_name, tform_map) in enumerate(transformation_maps['intensity based'].items()):
            tform_map_path = os.path.join(tform_output_folder, f"{idx+2}_intensity_based_{tform_name}_transformation_map.txt")
            with open(tform_map_path, "w") as f:
                for item in tform_map.GetParameterMap(0).items():
                    s = str(item).replace("'", "").replace(",", "")
                    f.write(f"{s}\n")

    elif adv_tform == 'feature':
        key = list(transformation_maps.keys())[1]
        adv_tform_map = transformation_maps[key]
        np.save(os.path.join(tform_output_folder, f"{2}_feature_based_{key}_transformation_map.npy"), adv_tform_map.params)

    print(f"Transformation maps saved to {tform_output_folder}")



@app.command(name="extract-channel")
def extract_channel_cmd(
    file_path: str = typer.Argument(..., help="Path to the input image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the image with extracted channel"),
    channel_idx: int = typer.Option(0, help="Channel index to extract (Default = 0 for DAPI)", show_default=True),
):
    """
    Extract a specific channel from a multiplexed image and save it as a separate image. And save the image with extracted channel to the specified output folder.

    Parameters:
    - file_path (str): Path to the input image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder_path (str): Folder to save the image with extracted channel
    - channel_idx (int, optional): Channel index to extract (default is 0 for DAPI)

    Returns:
     None. Saves the image with extracted channel to the specified output folder.
    """

    img = load_image_data(file_path)
    img_ch = extract_channel(img, channel_idx)

    img_folder_path = os.path.join(output_folder_path, "channel_extracted_image")
    os.makedirs(img_folder_path, exist_ok=True)
    img_path = os.path.join(img_folder_path, f"multiplexed_channel_{channel_idx}.tif")
    imwrite(img_path, img_ch)
    print(f"Image with extracted channel saved to {img_path}")



@app.command(name="transform-seg-mask")
def transform_seg_mask_cmd(
    mask_path: str = typer.Argument(..., help="Path to the segmentation mask of the moving image (.npy)"),
    fixed_path: str = typer.Argument(..., help="Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)"),
    output_folder_path: str = typer.Argument(..., help="Folder to save the transformed segmentation mask"),
    tform_map_path: str = typer.Argument(..., help="Path to the transformation maps folder"),
    moving_px_sz: str = typer.Argument(..., help="Path to moving image if .ome.tiff or Pixel size of the moving image"),
    fixed_px_sz: float = typer.Option(None, help="Pixel size of the fixed image (if image is not .ome.tif)", show_default=True)
):
    """
    Transform a segmentation mask of the moving image using the provided transformation maps and save the transformed segmentation mask to the specified output folder.

    Parameters:
    - mask_path (str): Path to the segmentation mask of the moving image (.npy)
    - fixed_path (str): Path to the fixed image (.tif/.tiff/.ome.tif/.ome.tiff)
    - output_folder_path (str): Folder to save the transformed segmentation mask    
    - tform_map_path (str): Path to the transformation maps folder
    - moving_px_sz (str): Path to moving image if .ome.tiff or Pixel size of the moving image
    - fixed_px_sz (float, optional): Pixel size of the fixed image (if image is not .ome.tif)

    Returns:
     None. Saves the transformed segmentation mask to the specified output folder.
    """

    # load mask
    mask = np.load(mask_path) # will need to change according to mask format
    print(f"Loaded segmentation mask.")

    # load and create transformation parameter objects
    transformation_maps = {}

    tform_files = sorted(os.listdir(tform_map_path), key=lambda x: int(x.split("_")[0]))
    transformation_maps['initial similarity'] = AffineTransform(matrix=np.load(os.path.join(tform_map_path, tform_files[0])))

    if len(tform_files) >= 2:
        second_file = tform_files[1]
        if "feature" in second_file:
            if "affine" in second_file:
                transformation_maps['affine'] = AffineTransform(matrix=np.load(os.path.join(tform_map_path, second_file)))
            else:
                transformation_maps[second_file.split("_")[3]] = ProjectiveTransform(matrix=np.load(os.path.join(tform_map_path, second_file)))

        elif "intensity" in second_file:
            print("Transforming segmentation mask using intensity based transformation maps with CLI is not yet supported. This functionality is possible in python package usage.")

            #intensity_tform_maps = {}

            #for file in tform_files[1:]:
            #    reg_map = itk.ParameterObject.New()
            #    reg_map.AddParameterFile(str(os.path.join(tform_map_path, file)))
                
            #    intensity_tform_maps[file.split("_")[3]] = reg_map

            #transformation_maps['intensity based'] = intensity_tform_maps

    print("Loaded transformation maps.")

    fixed_init = load_image_data(fixed_path)

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")
    

    try:
        moving_px_sz, _ = get_pixel_size_ome_tiff(moving_px_sz)
    except:
        pass

    try:
        scale = float(moving_px_sz) / fixed_px_sz
    except:
        raise ValueError("Could not determine moving image pixel sizes for scaling. Please check the provided pixel size or moving image path (ome.tiff).")
    

    if len(fixed_init.shape) == 2:
        fixed_init_sc = resize(fixed_init, (int(fixed_init.shape[0]/scale), int(fixed_init.shape[1]/scale)), anti_aliasing=True)
    else:
        fixed_init_sc = resize(fixed_init, (int(fixed_init[:, :, 0].shape[0]/scale), int(fixed_init[:, :, 0].shape[1]/scale)), anti_aliasing=True)
    fixed_img_shape = (int(fixed_init_sc.shape[0]), int(fixed_init_sc.shape[1]))

    moved_mask = transform_seg_mask(mask, transformation_maps, output_shape=fixed_img_shape, mpp=moving_px_sz)

    os.makedirs(output_folder_path, exist_ok=True)
    np.save(os.path.join(output_folder_path, "transformed_segmentation_mask.npy"), moved_mask)
    print(f"Transformed segmentation mask saved to {output_folder_path}/transformed_segmentation_mask.npy")



def main():
    app()


if __name__ == "__main__":
    main()