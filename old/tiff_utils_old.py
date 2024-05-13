import glob
from PIL import Image
import os
import numpy as np
from osgeo import gdal
import numpy as np
from PIL import Image

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

def reassemble_patches_to_tiff(patch_paths, patch_width, patch_height, original_dims, output_tiff_path, thresh):
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
    # combined_width, combined_height = patch_width * 21, patch_height * 21
    
    # Create a new, blank image with the dimensions of the original image
    reassembled_img = Image.new('F', (original_width, original_height), 1)
    # reassembled_img = Image.new('F', (combined_width, combined_height), 1)
    
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
            raise Exception("Invalid file format, expecting .TIF file.")
                            
    # Binarize the array based on a threshold
    reassembled_img = imbinarize(reassembled_img, thresh)
    
    # width_padding = (combined_width - original_width) // 2
    # height_padding = (combined_height - original_height) // 2
    
    # left = width_padding
    # upper = height_padding
    # right = width_padding + original_width
    # lower = height_padding + original_height
    
    # # Crop the image
    # reassembled_img = reassembled_img.crop((left, upper, right, lower))
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


# Note: Function calls are commented out to prevent execution. To use these functions, you need to specify
# the required parameters and uncomment the calls.

# Example of how to use these functions:
if __name__ == '__main__':
    _38_CLOUD_ROOT = "/mnt/netstore1_home/wyatt.mccurdy/NASA-Cloud-Detection-Data/38-Cloud"
    scene_path = _38_CLOUD_ROOT + '/38-Cloud_test/Entire_scene_gts/edited_corrected_gts_LC08_L1TP_050024_20160520_20170324_01_T1.TIF'
    img = Image.open(scene_path)
    original_width = img.size[0]
    original_height = img.size[1]
    patch_width = 384
    patch_height = 384
    thresh = 12 / 255

    split_result = split_tiff_into_patches(scene_path, patch_width, patch_height, './scene_patches')
    # reassembled_image_path = reassemble_patches_to_tiff('./sample_LC08_L1TP_050024_20160520_20170324_01_T1', patch_width, patch_height, (original_width, original_height), './scene2_pred_combined.tiff', binarize=True, thresh=thresh)

    # img = Image.open('./edited_corrected_gts_LC08_L1TP_064015_20160420_20170223_01_T1.TIF')
    # img.convert('F')
    # img.save('f_converted_image.tiff')
    # pad_horizontal = (original_width, 22 * patch_height - original_height)
    # pad_vertical = (22 * patch_width - original_width, original_height)
    # print(pad_horizontal)
    # print(pad_vertical)
    
    # padding_horizontal = Image.new('F', pad_horizontal, 0) 
    # padding_vertical = Image.new('F', pad_vertical, 0)

    # print(img.size)
    # print(img.mode)

    # img.paste(pad_vertical)
    # img.paste(pad_horizontal)

    # img.save('padded_image.tiff')
    # for item in os.listdir('./sample_LC08_L1TP_050024_20160520_20170324_01_T1'):
    #     if item.endswith('.TIF'):
    
def reassemble_patches_to_tiff2(patch_paths, patch_width, patch_height, original_dims, output_tiff_path, thresh=12 / 255):
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