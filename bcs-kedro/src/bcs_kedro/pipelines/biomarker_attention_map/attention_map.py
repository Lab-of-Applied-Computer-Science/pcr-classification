import logging
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import sparse
import cv2
from sklearn.neighbors import KernelDensity
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


# Detects if the mask is completely black (has no cell detection)
def is_empty_mask(mask):
    total_sum = mask.sum()
    logger.debug("Total sum %f", total_sum)
    return total_sum == 0.0


def generate_attention_map_with_tumor_cell(
    segmented_image: np.ndarray,
    deconv_img_ki67: np.ndarray,
    deconv_img_phh3: np.ndarray,
    tile_size=(1000, 1000),
    orig_img_size=(8000, 8000),
    kde_samples=100,
    bandwidth=150,
):

    logger.debug("Initializing final mask and cells for the KDE fit")
    finalmask = np.zeros(orig_img_size)
    centroids_x = []
    centroids_y = []

    logger.info("Transforming Ki67 mask's white pixel from 255 to 1")
    deconv_img_ki67[deconv_img_ki67 > 0] = 1
    logger.info("Transforming PHH3 mask's white pixel from 255 to 1")
    deconv_img_phh3[deconv_img_phh3 > 0] = 1

    num_tiles_x = orig_img_size[0] // tile_size[0]
    num_tiles_y = orig_img_size[1] // tile_size[1]

    logger.debug("Iterating over tiles from segmented image")
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            logger.debug("Retrieving masks for tile (%d, %d)", i, j)
            masks = segmented_image[i][j]

            if masks.size == 0:
                continue

            subki = deconv_img_ki67[
                i * tile_size[0] : i * tile_size[0] + tile_size[0],
                j * tile_size[1] : j * tile_size[1] + tile_size[1],
            ]
            subphh = deconv_img_phh3[
                i * tile_size[0] : i * tile_size[0] + tile_size[0],
                j * tile_size[1] : j * tile_size[1] + tile_size[1],
            ]

            x0 = i * tile_size[0]
            y0 = j * tile_size[1]

            logger.debug("x0: %d | y0: %d", x0, y0)

            for k in range(masks.shape[-1]):
                logger.debug("Processing mask %d", k + 1)
                mask = masks[:, :, k]
                logger.debug("Mask shape: %s", mask.shape)
                if not is_empty_mask(mask):
                    logger.debug("Mask: %s", mask)
                    # Masks can assume values of 255
                    white_x_coords, white_y_coords = np.where(mask > 0)
                    logger.debug(
                        "White X Coordinates: %s | White Y Coordinates: %s",
                        white_x_coords,
                        white_y_coords,
                    )

                    logger.debug("Calculating centroids")
                    centroid_x = np.mean(white_x_coords)
                    centroid_y = np.mean(white_y_coords)
                    logger.debug(
                        "Centroid coordinates: (%f, %f)", centroid_x, centroid_y
                    )
                    # TODO: revisitar essa parte por conta dos erros com aglomerações de células no algoritmo segmentação

                    if (subki[int(centroid_x), int(centroid_y)].item() > 0) or (
                        subphh[int(centroid_x), int(centroid_y)].item() > 0
                    ):
                        centroids_x.append(centroid_x + x0)
                        centroids_y.append(centroid_y + y0)
                else:
                    logger.debug("Mask is empty. Skipping to the next mask.")

    if not centroids_x:
        return finalmask

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


def generate_attention_map(
    deconv_img_ki67: np.ndarray,
    deconv_img_phh3: np.ndarray,
    orig_img_size=(8000, 8000),
    bandwidth=150,
    subsample_stride=5,
    kde_samples=50,
    flood_diff=50,
    density_gamma=0.4, 
    kernel_type='exponential'
):
    
    morph_ops = [
    ('close', 8, 5),  # Wide closure first
    ('open', 6, 3)    # Clean small gaps
    ]
    
    # combined_mask = deconv_img_ki67 > 0
    # combined_mask = deconv_img_phh3 > 0
    # combined_mask = np.logical_and(deconv_img_ki67 > 0,deconv_img_phh3 > 0) 
    
    combined_mask = np.logical_or(deconv_img_ki67 > 0, deconv_img_phh3 > 0)
    combined_8u = (combined_mask.astype(np.uint8) * 255)
    combined_8u = np.squeeze(combined_8u)
    
    print(f"deconv_img_ki67 shape: {deconv_img_ki67.shape}")
    print(f"deconv_img_phh3 shape: {deconv_img_phh3.shape}")
    print(f"combined_8u shape: {combined_8u.shape}")
    
    h, w = combined_8u.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    corners = [(0,0), (w-1, 0), (0,h-1), (w-1,h-1)]
    
    for (x,y) in corners:
        if combined_8u[y, x] == 255:
            cv2.floodFill(
                combined_8u, flood_mask,
                seedPoint=(x,y),
                newVal=0,
                loDiff=flood_diff,
                upDiff=flood_diff,
                flags=cv2.FLOODFILL_FIXED_RANGE | 4
            )
    
    
    for operation, kernel_size, iterations in morph_ops:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        if operation == 'close':
            cv2.morphologyEx(
                combined_8u, cv2.MORPH_CLOSE,
                kernel, combined_8u,
                iterations=iterations
            )
        elif operation == 'open':
            cv2.morphologyEx(
                combined_8u, cv2.MORPH_OPEN,
                kernel, combined_8u,
                iterations=iterations
            )

    # Optional: Save intermediate image for debugging
    cv2.imwrite("post_morphology.jpg", combined_8u)
    
    # Subsample coordinates
    combined_mask = combined_8u > 0
    yy, xx = np.where(combined_mask)
    subsampled_xx = xx[::subsample_stride]
    subsampled_yy = yy[::subsample_stride]
    
    if len(subsampled_xx) == 0:
        return np.zeros(orig_img_size)
    
    # Create grid and fit KDE with enhanced kernel
    x_grid = np.arange(0, orig_img_size[1], kde_samples)
    y_grid = np.arange(0, orig_img_size[0], kde_samples)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    
    kde = KernelDensity(
        bandwidth=bandwidth,
        kernel=kernel_type  # Changed to exponential for longer tails
    )
    kde.fit(np.vstack([subsampled_xx, subsampled_yy]).T)
    
    # Compute and shape density
    xy_eval = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    log_density = kde.score_samples(xy_eval)
    low_res_map = np.exp(log_density).reshape(grid_x.shape)
    
    # Apply non-linear compression to extend tails
    low_res_map = low_res_map ** density_gamma  # Key line for slower decay
    
    # Interpolate and normalize
    interp_fn = RectBivariateSpline(y_grid, x_grid, low_res_map)
    attention_map = interp_fn(np.arange(orig_img_size[0]), np.arange(orig_img_size[1]))
    
    # Normalization with softmax-like preservation of relative differences
    eps = 1e-8
    attention_map = (attention_map - attention_map.min() + eps) ** 0.75
    attention_map /= attention_map.max()
    
    return attention_map