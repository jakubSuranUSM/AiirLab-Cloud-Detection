"""
Cloud-Net Pipeline

Author: Jakub Suran
Date: March 31, 2024
Team: AiirLab
Institution: University of Southern Maine

This file contains the complete pipeline for the evaluation of Cloud-Net on the 95-Cloud and L8 Biome datasets.
"""
import os
import shutil
import pandas as pd

from evaluation import calculate_metrics_for_all_scenes, extract_unique_sceneids, stitch_all_patches
from preprecessing import check_sum_CN_data, copy_38_Cloud_test_data, create_new_patch_csv, extract_L8_bands, extract_L8_masks
from CloudNet.predict import generate_predictions
from CloudNet.utils import get_input_image_names

def main():
    # temporarly working with 38 CLoud instead of 95 Cloud 
    # (model trained on 95 Cloud is not accessible and has to be trained manually)

    PROJECT_ROOT = "/mnt/netstore1_home/jakub.suran/Cloud-Net_evaluation"
    _38_CLOUD_ROOT = "/mnt/netstore1_home/wyatt.mccurdy/NASA-Cloud-Detection-Data/38-Cloud"
    LANDSAT8_BIOME_ROOT = "/mnt/netstore1_home/wyatt.mccurdy/NASA-Cloud-Detection-Data/Landsat8Biome/extracted_files"
    PATCH_WIDTH, PATCH_HEIGHT = 384, 384
        
    # download and unzip the 95 Cloud and L8 Biome datasets

    # create directories for the test data
    # copy 95 Cloud test data
    
    # print("Copying 38 Cloud test data...")
    # test_folder = os.path.join(_38_CLOUD_ROOT, "38-Cloud_test")
    # shutil.copytree(test_folder, PROJECT_ROOT)

    # prepare data - create patches from L8 Biome data
    GROUND_TRUTH_ROOT = os.path.join(PROJECT_ROOT, "Entire_scene_gts")
    # extract_L8_masks(LANDSAT8_BIOME_ROOT, GROUND_TRUTH_ROOT)
    # extract_L8_bands(LANDSAT8_BIOME_ROOT, PROJECT_ROOT, PATCH_WIDTH, PATCH_HEIGHT)
    # check_sum_CN_data(PROJECT_ROOT)

    csv_filename = "test_patches.csv"
    # create_new_patch_csv(PROJECT_ROOT, csv_filename)

    # load the model and create predictions on the test data
    in_rows = PATCH_HEIGHT
    in_cols = PATCH_WIDTH
    num_of_channels = 4
    num_of_classes = 1
    batch_sz = 10
    max_bit = 65535
    thresh = 12 / 255
    PRED_FOLDER = os.path.join(PROJECT_ROOT, 'predictions')
    FULL_SCENE_PRED_FOLDER = os.path.join(PROJECT_ROOT, 'full_scene_predictions')
    weights_path = os.path.join(PROJECT_ROOT, 'CloudNet.h5')

    # df_test_img = pd.read_csv(os.path.join(PROJECT_ROOT, csv_filename))
    # test_img, test_ids = get_input_image_names(df_test_img, PROJECT_ROOT, if_train=False)

    # generate_predictions((in_rows, in_cols, num_of_channels), num_of_classes, batch_sz, max_bit, PRED_FOLDER, weights_path, test_img, test_ids)

    # evaluate the model on the test data
    
    if not os.path.exists(FULL_SCENE_PRED_FOLDER):
        os.makedirs(FULL_SCENE_PRED_FOLDER)
    
    scene_ids = extract_unique_sceneids(PRED_FOLDER)
    stitch_all_patches(scene_ids, GROUND_TRUTH_ROOT, PRED_FOLDER, FULL_SCENE_PRED_FOLDER, thresh)

    metrics_path = os.path.join(PROJECT_ROOT, 'metrics.csv')
    calculate_metrics_for_all_scenes(scene_ids, metrics_path, GROUND_TRUTH_ROOT, FULL_SCENE_PRED_FOLDER, thresh)

    # display evaluation
    df_metrics = pd.read_csv(metrics_path)
    print("Evaluation:")
    print(df_metrics.head())
    
    
if __name__ == '__main__':
    main()
    