import os
import shutil

import pandas as pd
from tiff_utils import convert_envi_to_binary_tiff, split_tiff_into_patches
from tqdm import tqdm


def copy_38_Cloud_test_data(_38_CLOUD_ROOT, project_root):
    test_folder = os.path.join(_38_CLOUD_ROOT, "38-Cloud_test")
    shutil.copytree(test_folder, project_root)
    

def extract_L8_masks(L8_root, output_folder):
    scene_folders = os.listdir(L8_root)
    for scene_folder in tqdm(scene_folders, "Extracting L8 Biome masks..."):
        scene_path = os.path.join(L8_root, scene_folder)
        mask_path = os.path.join(scene_path, f"{scene_folder}_fixedmask.img")
        output_mask_path = os.path.join(output_folder, f"edited_corrected_gts_{scene_folder}.TIF")
        convert_envi_to_binary_tiff(mask_path, output_mask_path)


def extract_L8_bands(L8_root, output_folder, patch_width, patch_height):
    band_names = {
        'blue': 'B2',
        'green': 'B3',
        'red': 'B4',
        'nir': 'B5'
    }
    scene_folders = os.listdir(L8_root)
    for scene_folder in tqdm(scene_folders, "Extracting L8 Biome bands..."):
        scene_path = os.path.join(L8_root, scene_folder)
        for band_name, band_number in band_names.items():
            band_path = os.path.join(scene_path, f"{scene_folder}_{band_number}.TIF")
            band_dir = os.path.join(output_folder, f"test_{band_name}")
            split_tiff_into_patches(band_path, patch_width, patch_height, band_dir, band_name)


def preprocess_L8_biome():
    extract_L8_masks(LANDSAT8_BIOME_ROOT, CN_MASKS_ROOT)
    extract_L8_bands(LANDSAT8_BIOME_ROOT, CLOUD_NET_DATA_ROOT, PATCH_WIDTH, PATCH_HEIGHT)


def check_sum_CN_data(root_dir):
    print("Checking data...")
    scene_ids = set()
    band_names = ['blue', 'green', 'red', 'nir']
    number_of_patches = []
    
    for band_name in band_names:
        band_dir = os.path.join(root_dir, f"test_{band_name}")
        patch_files = os.listdir(band_dir)
        for patch_file in patch_files:
            scene_id = patch_file.split('_')[6:]
            scene_id = '_'.join(scene_id)
            scene_ids.add(scene_id)
        num_unique_scene_ids = len(scene_ids)
        scene_ids.clear()
        print(f"Number of unique scene IDs for band {band_name}: {num_unique_scene_ids}")
        number_of_patches.append(num_unique_scene_ids)
    
    mask_dir = os.path.join(root_dir, "Entire_scene_gts")
    mask_files = os.listdir(mask_dir)
    scene_list = []
    for mask_file in mask_files:
        scene_id = mask_file.split('_')[3:]
        scene_id = '_'.join(scene_id)
        scene_ids.add(scene_id)
        scene_list.append(scene_id)
    num_unique_scene_ids = len(scene_ids)
    print(f"Number of unique scene IDs for masks: {num_unique_scene_ids}")
    number_of_patches.append(num_unique_scene_ids)
    assert len(set(number_of_patches)) == 1, "Number of unique scenes in all test directories should be the same and should equal to number of unique ground truth scenes."
    print("Check complete...\n")


def create_new_patch_csv(root_dir, filename):
    print("Creating CSV file...")
    blue_dir = os.path.join(root_dir, "test_blue")
    patch_files = os.listdir(blue_dir)
    patches = [file.split('_', 1)[1] for file in patch_files]
    patches = [file.split('.')[0] for file in patches]
    df = pd.DataFrame(patches, columns=['name'])
    csv_path = os.path.join(root_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"CSV file created at: \n{csv_path}\n")


CLOUD_NET_DATA_ROOT = "/mnt/netstore1_home/jakub.suran/Cloud-Net_data" 
_38_CLOUD_ROOT = "/mnt/netstore1_home/wyatt.mccurdy/NASA-Cloud-Detection-Data/38-Cloud"
LANDSAT8_BIOME_ROOT = "/mnt/netstore1_home/wyatt.mccurdy/NASA-Cloud-Detection-Data/Landsat8Biome/extracted_files"
CN_MASKS_ROOT = os.path.join(CLOUD_NET_DATA_ROOT, "Entire_scene_gts")
PATCH_WIDTH, PATCH_HEIGHT = 384, 384

# copy_38_Cloud_test_data()
# preprocess_L8_biome()
# check_sum_CN_data(CLOUD_NET_DATA_ROOT)
# create_new_patch_csv(os.path.join(CLOUD_NET_DATA_ROOT, "test_patches.csv"))


