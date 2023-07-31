import json
import tarfile
import os
from PIL import Image
from tqdm import tqdm

# Path to your images and output tarfile
img_dir = 'CC3M_595K'
output_tarfile_prefix = 'CC3M_595K'

# Number of shards
num_shards = 8

# Open and parse your JSON file
with open('metadata.json', 'r') as f:
    data = json.load(f)

# Total number of samples
num_samples = len(data)

# Samples per shard
samples_per_shard = num_samples // num_shards

# Progress bar
pbar = tqdm(total=num_samples, desc='Processing samples')

for shard_id in range(num_shards):
    start_idx = shard_id * samples_per_shard
    end_idx = (shard_id + 1) * samples_per_shard if shard_id != num_shards - 1 else num_samples

    # Open the tarfile for writing
    with tarfile.open(f'{output_tarfile_prefix}_{shard_id}.tar', 'w') as tar:
        for i in range(start_idx, end_idx):
            # Get the id and caption from the JSON object
            id_ = data[i]['id']
            caption = data[i]['caption']

            # Construct the image filename and make sure it exists
            img_filename = os.path.join(img_dir, f'{id_}.jpg')
            if not os.path.exists(img_filename):
                print(f"Image file {img_filename} does not exist, skipping this entry.")
                continue

            # Open the image file and write it to the tarfile
            with open(img_filename, 'rb') as f:
                tarinfo = tarfile.TarInfo(name=f'{id_}.jpg')
                tarinfo.size = os.path.getsize(img_filename)
                tar.addfile(tarinfo, fileobj=f)

            # Write the caption to a new text file and add it to the tarfile
            with open(f'{id_}.txt', 'w') as f:
                f.write(caption)
            with open(f'{id_}.txt', 'rb') as f:
                tarinfo = tarfile.TarInfo(name=f'{id_}.txt')
                tarinfo.size = os.path.getsize(f'{id_}.txt')
                tar.addfile(tarinfo, fileobj=f)

            # Remove the temporary text file
            os.remove(f'{id_}.txt')

            # Update progress bar
            pbar.update(1)

# Close the progress bar
pbar.close()
