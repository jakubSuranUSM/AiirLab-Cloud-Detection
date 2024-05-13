import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from evaluation import get_patches_for_sceneid, extract_unique_sceneids
from tiff_utils import reassemble_band_patches, combine_rgb_bands

root = '/home/jakub.suran/netstore1/Cloud-Net_evaluation/'

def combine_all_band_patches():
    gt_dir = os.path.join(root, 'Entire_scene_gts')
    colors = ['blue', 'green', 'red']

    df = pd.read_csv('metrics.tsv', sep='\t')
    scene_ids = df['Scene ID']

    for color in colors:
        print("Processing band:", color)
        target_dir = os.path.join(root, f'full_scene_{color}')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        patch_dir = os.path.join(root, f'test_{color}')
        total_pathes = 0
        # Loop through each scene ID to stitch patches
        for scene_id in tqdm(scene_ids, "Combining pathes..."):
            gt_path = os.path.join(gt_dir, f'edited_corrected_gts_{scene_id}.TIF')
            gt = Image.open(gt_path)
            
            patches = get_patches_for_sceneid(patch_dir, scene_id)
            total_pathes += len(patches)
            
            # Reassemble patches to a single image
            reassemble_band_patches(patches, 384, 384, gt.size, f'{target_dir}/full_{color}_{scene_id}.TIF')

        print(f"Processed {total_pathes} patches.")
        
        
def combine_all_full_bands():    
    df = pd.read_csv('metrics.tsv', sep='\t')
    sorted_df = df.sort_values(by='F-1', ascending=False)
    sorted_df = sorted_df[['Scene ID', 'F-1', 'Accuracy']]
    sorted_df.reset_index(drop=True, inplace=True)
    
    target_dir = os.path.join(root, 'full_scene_rgb')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for idx, (scene_id, f1, accuracy) in tqdm(sorted_df.iterrows(), "Processing scenes...", total=len(sorted_df)):
        red_path = os.path.join(root, f'full_scene_red/full_red_{scene_id}.TIF')
        green_path = os.path.join(root, f'full_scene_green/full_green_{scene_id}.TIF')
        blue_path = os.path.join(root, f'full_scene_blue/full_blue_{scene_id}.TIF')
        combine_rgb_bands(red_path, green_path, blue_path, f'{target_dir}/{idx}_full_rgb_f1_{round(f1, 2)}acc_{round(accuracy, 2)}{scene_id}.TIF')
        
        
combine_all_full_bands()