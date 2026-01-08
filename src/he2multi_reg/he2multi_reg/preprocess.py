import histomicstk as htk
import numpy as np
from tifffile import imread, TiffFile, TiffWriter
from skimage.transform import resize
import xml.etree.ElementTree as ET



def colour_deconvolusion_preprocessing_HnE(hne_init):
    """
    Decompose HnE stained image into hematoxylin and eosin channels using color deconvolution and return the hematoxylin channel.

    Parameters:
    hne_init (ndarray): Input HnE stained image.

    Returns:
    hne_deconv (ndarray): Hematoxylin channel extracted from the HnE stained image.
    """

    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains

    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(hne_init, W)
    hne_deconv = 1 - imDeconvolved.Stains[:, :, 0]

    return hne_deconv




def get_image_size_ome_tiff(file_path):
    """
    Get the image size (height, width) from an OME-TIFF file.

    Parameters:
    - file_path (str) : path to the OME-TIFF file

    Returns:
    - shape (tuple) : (height, width) of the image
    """

    with TiffFile(file_path) as tif:
        img = tif.series[0].asarray()
        shape = img.shape[0:2] if img.ndim == 2 or img.shape[0] < img.shape[2] else img.shape[1:3] 
        return shape



def get_pixel_size_ome_tiff(file_path):
    """
    Get the pixel size (PhysicalSizeX, PhysicalSizeY) from an OME-TIFF file.

    Parameters:
    - file_path (str) : path to the OME-TIFF file

    Returns:
    - (px, py) (tuple) : PhysicalSizeX and PhysicalSizeY in micrometers/pixel
    """

    with TiffFile(file_path) as tif:
        ome = tif.ome_metadata
        if ome is None:
            raise ValueError(f"Not an OME-TIFF: {file_path}")

        root = ET.fromstring(ome)
        pixels = root.find(".//{*}Pixels")   

        px = pixels.get("PhysicalSizeX")
        py = pixels.get("PhysicalSizeY")

        px = float(px) if px is not None else None
        py = float(py) if py is not None else None

        return px, py



def load_image_data(file_path):
    if file_path.endswith(".tif") or file_path.endswith(".tiff"):
        img_raw = imread(file_path)
        img = np.array(img_raw) 

        return img if (len(img.shape) == 2) or (img.shape[2] < img.shape[0]) else img.transpose(1, 2, 0)
    
    else: 
        raise ValueError("Unsupported file format. Please provide a .tif file.")



def extract_channel(img, channel_index):
    
    return img[:, :, channel_index]



def load_and_scale_images(fixed_path, moving_path, fixed_px_sz, moving_px_sz):

    if fixed_px_sz is None:
        try:
            fixed_px_sz, _ = get_pixel_size_ome_tiff(fixed_path)
        except Exception:
            fixed_px_sz = None
        
        if fixed_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for fixed image. Please provide fixed_px_sz.")

    if moving_px_sz is None:
        try:
            moving_px_sz, _ = get_pixel_size_ome_tiff(moving_path)
        except Exception:
            moving_px_sz = None

        if moving_px_sz is None:
            raise ValueError("Pixel size information not found in metadata for moving image. Please provide moving_px_sz.")

    scale = moving_px_sz / fixed_px_sz

    # load fixed image
    fixed_img = load_image_data(fixed_path)
    if len(fixed_img.shape) == 2:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale)), anti_aliasing=True)
    elif fixed_img.shape[2] == 3:
        fixed_init = resize(fixed_img, (int(fixed_img.shape[0]/scale), int(fixed_img.shape[1]/scale), fixed_img.shape[2]), anti_aliasing=True)
    elif fixed_img.shape[2] > 3:
        fixed_ch_img = extract_channel(fixed_img, 0)
        fixed_init = resize(fixed_ch_img, (int(fixed_ch_img.shape[0]/scale), int(fixed_ch_img.shape[1]/scale)), anti_aliasing=True)
    fixed_init = fixed_init*255

    # load moving image
    moving_init = load_image_data(moving_path)

    return fixed_init, moving_init



def save_ome_tiff(
    img,
    out_path,
    channel_names=None,
    physical_size_x=None,
    physical_size_y=None,
    source_ome_xml=None):

    if img.ndim == 2:
        # grayscale (Y, X)
        Y, X = img.shape
        C = 1
        img = img.reshape(Y, X, 1)
        data = img.transpose(2, 0, 1)

    elif img.ndim == 3:
        if img.shape[2] == 3:
            # RGB (Y, X, 3)
            Y, X, C = img.shape
            data = img.transpose(2, 0, 1)

        elif img.shape[2] > 3:
            # multiplexed (Y, X, C)
            Y, X, C = img.shape
            data = img.transpose(2, 0, 1)

        else:
            raise ValueError(f"Unsupported shape {img.shape}: no alpha allowed and no Z/T.")

    else:
        raise ValueError(f"Unsupported ndim={img.ndim}")

    
    if channel_names is None:
        try:
            if source_ome_xml is not None:
                root = ET.fromstring(source_ome_xml)
                channel_names = [c.get("Name") for c in root.findall(".//{*}Channel")]
            else:
                # if not provided, auto-generate
                channel_names = [f"Channel_{i}" for i in range(C)]
        except:
            channel_names = [f"Channel_{i}" for i in range(C)]

    if len(channel_names) != C:
        raise ValueError(f"Channel name count {len(channel_names)} does not match C={C}")
    
    
    with TiffWriter(out_path, bigtiff=True) as tif:
         metadata={
             'axes': 'CYX',
             'PhysicalSizeX': physical_size_x,
             'PhysicalSizeXUnit': 'µm',
             'PhysicalSizeY': physical_size_y,
             'PhysicalSizeYUnit': 'µm',
             'Channel': {'Name': channel_names},
         }

         tif.write(
             data,
             resolution=(Y, X),
             metadata=metadata,
         )
