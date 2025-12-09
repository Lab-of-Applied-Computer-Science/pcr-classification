import os
import numpy as np
import cv2
from itertools import product
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def get_tiles(img: np.array, tile_width: int, tile_height: int):
    """Gets the tiles window and offset of an image

    Args:
        img (np.array): Image to be sliced.
        tile_width (int): Width of the tile.
        tile_height (int): Height of the tile.

    Yields:
        (tuple[int, int, numpy.array]): Tuple containing the
            column offset, row offset and the tile window.
    """
    img_width, img_height = img.shape[0], img.shape[1]
    tiles_offsets = product(
        range(0, img_width, tile_width), range(0, img_height, tile_height)
    )
    for col_offset, row_offset in tiles_offsets:
        tile_window = img[
            col_offset : (tile_width + col_offset),
            row_offset : (tile_height + row_offset),
        ]
        yield col_offset, row_offset, tile_window


def slice_image_into_tiles_disk(image_path, out_dir, tile_width, tile_height):
    os.makedirs(out_dir, exist_ok=True)
    tile_metadata = []
    img = cv2.imread(image_path)

    for col_offset, row_offset, tile in get_tiles(img, tile_width, tile_height):
        file_name_with_ext = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(file_name_with_ext)
        tile_file_name = f"{file_name}_tile-{tile_width}x{tile_height}_x0-{col_offset}_y0-{row_offset}"
        out_path = os.path.join(out_dir, f"{tile_file_name}{file_ext}")

        tile_metadata.append(
            {
                "tile_file_name": tile_file_name,
                "tile_img_path": out_path,
                "orig_img_width": img.shape[0],
                "orig_img_height": img.shape[1],
            }
        )
        print(f"Writing image to path: {out_path}")
        cv2.imwrite(out_path, tile)
    return tile_metadata


def slice_image_into_tiles_memory(
    img, base_output_path, out_file_ext, out_dir, tile_width, tile_height
):
    tile_metadata = []

    for col_offset, row_offset, tile in get_tiles(img, tile_width, tile_height):
        tile_file_name = base_output_path.replace(
            out_file_ext,
            f"_tile-{tile_width}x{tile_height}_x0-{col_offset}_y0-{row_offset}{out_file_ext}",
        )
        out_path = os.path.join(out_dir, tile_file_name)

        tile_metadata = {
            "tile_file_name": tile_file_name,
            "tile_img_path": os.path.abspath(out_path),
            "orig_img_width": img.shape[0],
            "orig_img_height": img.shape[1],
        }

        logger.debug(f"Writing image to path: {out_path}")
        yield tile, tile_metadata