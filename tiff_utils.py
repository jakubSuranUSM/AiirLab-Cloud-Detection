import glob
from PIL import Image
import os
import numpy as np
from osgeo import gdal


def split_tiff_into_patches(input_tiff_path, patch_width, patch_height, output_dir, band_color, padding_color=0):
    """
    Splits a TIFF image into patches of specified dimensions and saves them in the specified directory.
    Adds padding to the image if necessary to fit the patch dimensions.

    Parameters:
    - input_tiff_path: Path to the input TIFF image.
    - patch_width: Width of each patch.
    - patch_height: Height of each patch.
    - output_dir: Directory to save the patches.
    - padding_color: Color of the padding as a tuple (default is black).

    Returns:
    - A list of paths to the saved patch images.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the filename of the input tiff path
    filename = os.path.basename(input_tiff_path)
    # Extract sceneid from the filename
    sceneid = filename.split('_')[0]
    
    # Open the input image
    with Image.open(input_tiff_path) as img:
        # Calculate the required dimensions for padding
        padded_width = patch_width * ((img.width + patch_width - 1) // patch_width)
        padded_height = patch_height * ((img.height + patch_height - 1) // patch_height)
            
        width_padding = (padded_width - img.width) // 2
        height_padding = (padded_height - img.height) // 2
        
        # Create a new image with the padded dimensions and specified padding color
        padded_img = Image.new(img.mode, (padded_width, padded_height), padding_color)
        # Paste the original image onto the padded image
        padded_img.paste(img, (width_padding, height_padding))
        
        # Initialize a list to hold the paths of patch images
        patch_paths = []
        
        patch_number = 1
        # Extract and save patches from the padded image
        for i in range(padded_height // patch_height):
            for j in range(padded_width // patch_width):
                # Define the box to extract the patch
                box = (j * patch_width, i * patch_height, (j + 1) * patch_width, (i + 1) * patch_height)
                # Extract the patch
                patch = padded_img.crop(box)
                # Define patch file path
                patch_path = os.path.join(output_dir, f'{band_color}_patch_{patch_number}_{i+1}_by_{j+1}_{sceneid}.TIF')
                # Save the patch
                patch.save(patch_path)
                # Append the path to the list
                patch_paths.append(patch_path)
                patch_number += 1
                
    return patch_paths


def reassemble_patches_to_tiff(patch_paths, patch_width, patch_height, original_dims, output_tiff_path, thresh=12 / 255, parts=(2,4)):
    """
    Reassembles patches from a directory back into a TIFF image.

    Parameters:
    - patch_dir: Directory containing the patch images.
    - patch_width: Width of each patch.
    - patch_height: Height of each patch.
    - original_dims: A tuple of the original image's dimensions (width, height).
    - output_tiff_path: Path to save the reassembled TIFF image.

    Returns:
    - The path to the reassembled TIFF image.
    """
    original_width, original_height = original_dims
    combined_width = patch_width * (original_width // patch_width + 1)
    combined_height = patch_height * (original_height // patch_height + 1)
    
    assert combined_width >= original_width and combined_width - patch_width <= original_width
    assert combined_height >= original_height and combined_height - patch_height <= original_height
    
    # Create a new, blank image with the dimensions of the original image
    reassembled_img = Image.new('F', (combined_width, combined_height), 1)
    # print(reassembled_img.size)
    
    # Load and place each patch
    for patch_path in patch_paths:
        # Extract patch position from filename
        if patch_path.endswith('.TIF'):
            patch_filename = os.path.basename(patch_path)
            parts = patch_filename.split('_')
            i, j = int(parts[2]) - 1, int(parts[4]) - 1
            # Calculate patch's top-left corner in the reassembled image
            box = (j * patch_width, i * patch_height)
            # Open patch image
            with Image.open(patch_path) as patch_img:
                # Paste the patch into the reassembled image
                reassembled_img.paste(patch_img, box)
        else:
            raise Exception("Invalid file format.")
                            
    # Binarize the array based on a threshold
    reassembled_img = imbinarize(reassembled_img, thresh)
    
    width_padding = (combined_width - original_width) // 2
    height_padding = (combined_height - original_height) // 2
    
    # print("Width padding: ", width_padding)
    # print("Height padding: ", height_padding)
    
    left = width_padding
    upper = height_padding
    right = width_padding + original_width
    lower = height_padding + original_height
    
    # Crop the image
    reassembled_img = reassembled_img.crop((left, upper, right, lower))
    # Save the reassembled image
    reassembled_img.save(output_tiff_path)
    
    return output_tiff_path


def imbinarize(image, threshold):
    array = np.array(image)
    binarized_array = array > threshold
    binarized_img = Image.fromarray(binarized_array)
    return binarized_img


def convert_envi_to_binary_tiff(input_img_path, output_tiff_path):
    # Open the input ENVI .img file
    dataset = gdal.Open(input_img_path)
    if not dataset:
        raise IOError(f"Unable to open file {input_img_path}")
  
    # Read the first band of the image into a NumPy array
    band = dataset.GetRasterBand(1)
    img_array = band.ReadAsArray()

    # Convert the image to a binary mask:
    # Thin Cloud (192) and Cloud (255) are set to 255, everything else is set to 0
    binary_mask = np.where((img_array == 192) | (img_array == 255), 255, 0)
  
    img = Image.fromarray(binary_mask.astype(np.uint8))
    img = img.convert("1")
    img.save(output_tiff_path)

    print(f"Binary TIFF file created: {output_tiff_path}")


def reassemble_band_patches(patch_paths, patch_width, patch_height, original_dims, output_tiff_path, max_value=65535):
    """
    Reassembles band patches from a directory into one TIFF image.

    Parameters:
    - patch_dir: Directory containing the patch images.
    - patch_width: Width of each patch.
    - patch_height: Height of each patch.
    - original_dims: A tuple of the original image's dimensions (width, height).
    - output_tiff_path: Path to save the reassembled TIFF image.

    Returns:
    - The path to the reassembled TIFF image.
    """
    original_width, original_height = original_dims
    combined_width = patch_width * (original_width // patch_width + 1)
    combined_height = patch_height * (original_height // patch_height + 1)
    
    assert combined_width >= original_width and combined_width - patch_width <= original_width
    assert combined_height >= original_height and combined_height - patch_height <= original_height
    
    # Create a new, blank image with the dimensions of the original image
    reassembled_img = Image.new('L', (combined_width, combined_height), 1)
    # print(reassembled_img.size)
    
    # Load and place each patch
    for patch_path in patch_paths:
        # Extract patch position from filename
        if patch_path.endswith('.TIF'):
            patch_filename = os.path.basename(patch_path)
            parts = patch_filename.split('_')
            i, j = int(parts[3]) - 1, int(parts[5]) - 1
            # Calculate patch's top-left corner in the reassembled image
            box = (j * patch_width, i * patch_height)
            # Open patch image
            with Image.open(patch_path) as patch_img:
                patch_arr = np.array(patch_img)
                # normalize the array
                normalized_array = (patch_arr / max_value) * 255
                normalized_array = normalized_array.astype(np.uint8)

                patch_img = Image.fromarray(normalized_array)
                # Paste the patch into the reassembled image
                reassembled_img.paste(patch_img, box)
        else:
            raise Exception("Invalid file format.")
                            
    width_padding = (combined_width - original_width) // 2
    height_padding = (combined_height - original_height) // 2
    
    left = width_padding
    upper = height_padding
    right = width_padding + original_width
    lower = height_padding + original_height
    
    # Crop the image
    reassembled_img = reassembled_img.crop((left, upper, right, lower))
    # Save the reassembled image
    reassembled_img.save(output_tiff_path)
    
    return output_tiff_path


def combine_rgb_bands(red_band_path, green_band_path, blue_band_path, output_tiff_path):
    """
    Combines three TIFF images representing the red, green, and blue bands into one RGB TIFF image.

    Parameters:
    - red_band_path: Path to the TIFF image representing the red band.
    - green_band_path: Path to the TIFF image representing the green band.
    - blue_band_path: Path to the TIFF image representing the blue band.
    - output_tiff_path: Path to save the combined RGB TIFF image.

    Returns:
    - The path to the combined RGB TIFF image.
    """
    # Open the red, green, and blue band images
    with Image.open(red_band_path) as red_band, Image.open(green_band_path) as green_band, Image.open(blue_band_path) as blue_band:
        # Check if the bands have the same dimensions
        if red_band.size != green_band.size or red_band.size != blue_band.size:
            raise ValueError("The red, green, and blue bands must have the same dimensions.")
        
        # Convert the red, green, and blue bands to NumPy arrays
        red_array = np.array(red_band)
        green_array = np.array(green_band)
        blue_array = np.array(blue_band)

        # Stack the arrays along the third axis to create a 3D array
        rgb_array = np.stack((red_array, green_array, blue_array), axis=2)

        # Create an RGB image from the normalized array
        rgb_image = Image.fromarray(rgb_array, mode='RGB')
        
        # Save the RGB image
        rgb_image.save(output_tiff_path)
    return output_tiff_path

