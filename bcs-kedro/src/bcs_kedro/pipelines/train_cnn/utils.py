import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def reconstruct_image(
    ordered_image_list,
    tile_width,
    tile_height,
    orig_img_width,
    orig_img_height,
    nchannels=1,
):
    """
    Reconstructs a single image from tiles.

    Args:
        ordered_image_list (list): Ordered image list of tile image_paths.
        tile_width (int): Width of each tile in pixels.
        tile_height (int): Height of each tile in pixels.
        orig_img_width (int): Width of the original image.
        orig_img_height (int): Height of the original image.
        nchannels (int): Number of channels in the image.

    Returns:
        (np.ndarray): Reconstructed IHC image.
    """
    logger.debug("Reading image files as a flat array")
    flatten_img = np.array(
        [Image.open(img_tile_path).convert("L") for img_tile_path in ordered_image_list]
    )
    logger.debug("Flatten image vector shape %s", flatten_img.shape)

    num_tiles_per_row = orig_img_width // tile_width
    num_tiles_per_col = orig_img_height // tile_height

    reconstructed_img = flatten_img.reshape(
        num_tiles_per_row, num_tiles_per_col, tile_width, tile_height, nchannels
    )

    return np.concatenate(
        [
            np.concatenate(reconstructed_img[i], axis=1)
            for i in range(num_tiles_per_row)
        ],
        axis=0,
    )