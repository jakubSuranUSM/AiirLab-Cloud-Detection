# Cloud-Net pipeline

The Cloud-Net pipeline represents the evaluation process of the Cloud-Net model.
The pipeline will use both L8 Biome and 38-Cloud datasets for evaluation.

### Landsat 8 Spectral Ranges:<br>

| Band # | Name  | Spectral Range (nm) |
| ------ | ----- | ------------------- |
| 2      | Blue  | 450-515             |
| 3      | Green | 520-600             |
| 4      | Red   | 630-680             |
| 5      | NIR   | 845-885             |

## Tasks

- Preprocess L8 Biome data
  - Convert ENGI cloud masks (as .img files) to binary TIF
  - Cut out Bands 2-5 to 384x384 patches
- Combine 38-Cloud and L8 Biome into one dataset
- Make predictions on this dataset
- Calculate the metrics for the predictions and ground truths
  - Combine patches into full scene predictions
  - Calculate evaluation metrics on the full scenes

## Run

- Pull the CloudNet repository from `https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection` and place it in the `CloudNet` directory in the root directory

* run the `main.py` file to run the while pipeline (preprocessing, prediction, evaluation), or comment out code blocks to perform only certain tasks
