import glob
from pathlib import Path
import numpy as np
import tifffile as tifff
from skimage.util import img_as_float32
import itk
import os


def apply_transform(moving_img_itk, transform_map):
    
    no_of_transforms = transform_map.GetNumberOfParameterMaps()

    transformed_img = []
    result = moving_img_itk
    for n in range(no_of_transforms):
        single_transform = itk.ParameterObject.New()
        single_transform.AddParameterMap(transform_map.GetParameterMap(n))
        result = itk.transformix_filter(result, single_transform, log_to_console=False)
        transformed_img.append(itk.GetArrayFromImage(result))

    return transformed_img


def register_references(fixed, moving, mpp, results_dir, transformation):
    
    if transformation == 'rigid':
        transform_scheme = ["01_Rigid"]
    elif transformation == 'affine':
        transform_scheme = ["01_Rigid", "02_Affine"]
    elif transformation == 'bspline':
        transform_scheme = ["01_Rigid", "02_Affine", "03_BSpline"]

    global_trf_map = itk.ParameterObject.New()
    workdir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    registration_maps = sorted(glob.glob(os.path.join(workdir, "transformation_parameters", "*.txt")))
    out_transform_dir = os.path.join(results_dir, "transforms")

    qc_out=[]
    for element in transform_scheme:
        aux = os.path.join(out_transform_dir, element)
        os.makedirs(aux, exist_ok=True)
        qc_out.append(aux)

    fixed_itk = itk.GetImageFromArray(img_as_float32(fixed))
    fixed_itk.SetSpacing([mpp,mpp])

    moving_itk = itk.GetImageFromArray(img_as_float32(moving))
    moving_itk.SetSpacing([mpp,mpp])

    for Reg, Out in zip(registration_maps, qc_out):
        reg_map = itk.ParameterObject.New()
        reg_map.AddParameterFile(str(Reg))

        moving_itk, result_trf_params = itk.elastix_registration_method(
            fixed_itk, 
            moving_itk,
            parameter_object = reg_map,
            output_directory = str(Out),
            log_file_name = "log.txt",
            log_to_console =False
        )

        global_trf_map.AddParameterMap(result_trf_params.GetParameterMap(0))

    registered_imgs = apply_transform(moving_itk, global_trf_map)

    return global_trf_map, registered_imgs


def registration_with_intensity(dapi_img, hne_img, mpp, output_dir, transformation='bspline'):

    output_subdir = os.path.join(output_dir, "results_registration_with_intensity")
    os.makedirs(output_subdir, exist_ok=True)

    #Extract transforms (transformation map) 
    transformations_map, registered_imgs = register_references(
        dapi_img,
        hne_img,
        mpp,
        output_subdir,
        transformation
        )

    return transformations_map, registered_imgs











