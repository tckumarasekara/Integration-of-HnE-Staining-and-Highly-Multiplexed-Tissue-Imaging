from tifffile import imread
import numpy as np
import os

def load_HnE_image(folder_path):

    img_data = []
    label_data = []
    
    for _, _, files in os.walk(folder_path):
        for index, file in enumerate(files):
            if file.endswith(".tif") and index < 100: # change later
                image_path = os.path.join(folder_path, file)
                img_raw = imread(image_path)
                #print(f"Loaded image shape: {np.array(img_raw).shape}")
                img = np.array(img_raw)[0:3, :, :]
                label = np.array(img_raw)[3, :, :]
    
                img_data.append(img)
                label_data.append(label)

    return img_data, label_data
