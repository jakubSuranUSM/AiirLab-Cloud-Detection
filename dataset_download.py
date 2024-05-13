# Define the URLs for the zip files
url_38_cloud = "https://vault.sfu.ca/index.php/s/pymNqYF09JkM8Bp/download"
url_l8_biome = "https://example.com/l8_biome.zip"

# Define the directory to download and unzip the files
data_dir = "/home/jakub.suran/Documents/Cloud-Net_pipeline/data"

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download and unzip the 95 Cloud dataset
zip_file_38_cloud = os.path.join(data_dir, "38_cloud.zip")
response_38_cloud = requests.get(url_38_cloud, stream=True)

# Get the file size for the progress bar
file_size_38_cloud = int(response_38_cloud.headers.get("Content-Length", 0))

# Create the progress bar
progress_bar_38_cloud = tqdm(total=file_size_38_cloud, unit="B", unit_scale=True)

with open(zip_file_38_cloud, "wb") as file:
    for data in response_38_cloud.iter_content(chunk_size=file_size_38_cloud // 1000):
        # Update the progress bar
        progress_bar_38_cloud.update(len(data))
        file.write(data)

# Close the progress bar
progress_bar_38_cloud.close()

# Unzip the 95 Cloud dataset
with zipfile.ZipFile(zip_file_38_cloud, "r") as zip_ref:
    zip_ref.extractall(data_dir)

# Delete the zip file
os.remove(zip_file_38_cloud)
"""
# Download and unzip the L8 Biome dataset
zip_file_l8_biome = os.path.join(data_dir, "l8_biome.zip")
response_l8_biome = requests.get(url_l8_biome, stream=True)

# Get the file size for the progress bar
file_size_l8_biome = int(response_l8_biome.headers.get("Content-Length", 0))

# Create the progress bar
progress_bar_l8_biome = tqdm(total=file_size_l8_biome, unit="B", unit_scale=True)

with open(zip_file_l8_biome, "wb") as file:
    for data in response_l8_biome.iter_content(chunk_size=1024):
        # Update the progress bar
        progress_bar_l8_biome.update(len(data))
        file.write(data)

# Close the progress bar
progress_bar_l8_biome.close()

# Unzip the L8 Biome dataset
with zipfile.ZipFile(zip_file_l8_biome, "r") as zip_ref:
    zip_ref.extractall(data_dir)

# Delete the zip file
os.remove(zip_file_l8_biome)
"""
