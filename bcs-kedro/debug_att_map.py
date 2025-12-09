import logging
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Static Parameters
PARAMS = {
    "src_deconv_metadata": "/home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/bcs-kedro/data/03_primary/spatial-attention-pcr-deconv/metadata.parquet",
    "src_segmented_metadata": "/home_cerberus/disk3/speed/viniciusfariaresende/breast-cancer-segmentation-storage/individaully_segmented_masks/metadata.parquet",
    "output_dir": "/home_cerberus/speed/henrique.colonese/full_pipeline/breast-cancer-segmentation/bcs-kedro/debug_att_map",
    "num_images_to_process": 10,  # Number of images to process
    "tile_width": 1000,
    "tile_height": 1000,
    "orig_img_width": 8000,
    "orig_img_height": 8000,
    "kde_samples": 512,
    "bandwidth": 150,
}

def is_empty_mask(mask):
    total_sum = mask.sum()
    logger.debug("Total sum %f", total_sum)
    return total_sum == 0.0

def load_and_merge_metadata():
    # Load metadata from both Parquet files
    deconv_metadata = pd.read_parquet(PARAMS['src_deconv_metadata'])
    segmented_metadata = pd.read_parquet(PARAMS['src_segmented_metadata'])

    # Separate paths for KI67 and PHH3
    ki67_metadata = deconv_metadata[deconv_metadata['biomarker'] == 'KI67']
    phh3_metadata = deconv_metadata[deconv_metadata['biomarker'] == 'PHH3']

    # Merge dataframes on 'img_id'
    merged_metadata = pd.merge(
        segmented_metadata,
        ki67_metadata[['img_id', 'path']],
        on='img_id',
        how='inner',
        suffixes=('', '_ki67')
    ).merge(
        phh3_metadata[['img_id', 'path']],
        on='img_id',
        how='inner',
        suffixes=('', '_phh3')
    )

    return merged_metadata

def select_and_verify_images(metadata):
    # Choose a subset of 'img_id' entries
    sample_metadata = metadata.sample(PARAMS['num_images_to_process'], random_state=42)

    images_data = []

    for index, row in sample_metadata.iterrows():
        # Gather paths
        he_image_path = row['orig_img_path']
        print(he_image_path)
        segmented_image_path = row['path']
        print(segmented_image_path)
        deconv_img_ki67_path = row['path_ki67']
        print(deconv_img_ki67_path)
        deconv_img_phh3_path = row['path_phh3']
        print(deconv_img_phh3_path)

        # Verify existence
        if all(os.path.exists(p) for p in [he_image_path, segmented_image_path, deconv_img_ki67_path, deconv_img_phh3_path]):
            images_data.append({
                'he_image_path': he_image_path,
                'segmented_image_path': segmented_image_path,
                'deconv_img_ki67_path': deconv_img_ki67_path,
                'deconv_img_phh3_path': deconv_img_phh3_path
            })
        else:
            logger.warning(f"Missing files for img_id {row['img_id']}")

    return images_data

def save_results(output_dir, he_image, seg_overlay, deconv_img_ki67, deconv_img_phh3, attention_map):
    os.makedirs(output_dir, exist_ok=True)

    # Save Original HE Image
    plt.imsave(os.path.join(output_dir, "he_image.png"), he_image)

    # Save HE with Segmented Overlay
    plt.imsave(os.path.join(output_dir, "segmented_overlay.png"), seg_overlay)

    # Save Deconvolved Images
    plt.imsave(os.path.join(output_dir, "deconv_ki67.png"), deconv_img_ki67, cmap='gray')
    plt.imsave(os.path.join(output_dir, "deconv_phh3.png"), deconv_img_phh3, cmap='gray')

    # Save Heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap (KDE of Centroid Activity)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "attention_map.png"))
    plt.close()

def process_npz_file(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        # Debug step: print out keys in the npz file to check its content
        logger.debug(f"Keys in npz: {list(data.keys())}")
        
        # Assuming the npz file contains a key 'masks' or something similar; adjust this to your actual key
        try:
            masks = data['contours']  # Replace 'masks' with the appropriate key
        except KeyError:
            logger.error("Key 'contours' not found in the npz file.")
            raise
        
        return masks

def generate_attention_map(contours, deconv_img_ki67, deconv_img_phh3, tile_size, orig_img_size, kde_samples, bandwidth):
    logger.debug("Initializing final mask and cells for the KDE fit")
    finalmask = np.zeros(orig_img_size)
    centroids_x = []
    centroids_y = []
    logger.debug("Transforming Ki67 mask's white pixel from 255 to 1")
    deconv_img_ki67[deconv_img_ki67 > 0] = 1
    logger.debug("Transforming PHH3 mask's white pixel from 255 to 1")
    deconv_img_phh3[deconv_img_phh3 > 0] = 1

    logger.debug("Processing contours")
    for contour in contours:  # Assuming contours is a list of arrays

        mask = np.zeros(orig_img_size, dtype=np.uint8)
        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Calculate centroid
        if not is_empty_mask(mask):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Check if this centroid falls within the area of deconvolution images
                if (deconv_img_ki67[centroid_y, centroid_x] > 0) or (deconv_img_phh3[centroid_y, centroid_x] > 0):
                    centroids_x.append(centroid_x)
                    centroids_y.append(centroid_y)

    if not centroids_x:
        return finalmask

    # Convert centroids into a numpy array for KDE processing
    xy_train = np.vstack([centroids_x, centroids_y]).T
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(xy_train)
    
    step_x, step_y = (
        orig_img_size[0] // kde_samples,
        orig_img_size[1] // kde_samples,
    )
    grid_xs, grid_ys = np.mgrid[
        0 : orig_img_size[0] : step_x, 0 : orig_img_size[1] : step_y
    ]
    xy_eval = np.vstack([grid_xs.flatten(), grid_ys.flatten()]).T
    prob = np.exp(kde_skl.score_samples(xy_eval))
    attention_map = np.reshape(prob, grid_xs.shape)

    return attention_map

def main():
    # Load and merge metadata
    metadata = load_and_merge_metadata()
    # Select and verify images
    images_data = select_and_verify_images(metadata)
    # Process images
    for data in images_data:
        he_image = cv2.imread(data['he_image_path'])
        contours = process_npz_file(data['segmented_image_path'])  # Updated to use contours
        deconv_img_ki67 = cv2.imread(data['deconv_img_ki67_path'], cv2.IMREAD_ANYDEPTH)
        deconv_img_phh3 = cv2.imread(data['deconv_img_phh3_path'], cv2.IMREAD_ANYDEPTH)
        # Generate attention map
        attention_map = generate_attention_map(
            contours, deconv_img_ki67, deconv_img_phh3,
            (PARAMS['tile_width'], PARAMS['tile_height']), (PARAMS['orig_img_width'], PARAMS['orig_img_height']),
            PARAMS['kde_samples'], PARAMS['bandwidth']
        )
        # Overlay Segmented Cells
        seg_overlay = he_image.copy()
        cv2.drawContours(seg_overlay, contours, -1, (0, 255, 0), 2)
        # Save Results
        save_results(PARAMS['output_dir'], he_image, seg_overlay, deconv_img_ki67, deconv_img_phh3, attention_map)

if __name__ == "__main__":
    main()

