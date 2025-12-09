"""
    Module containing all nodes that are part of the attention map generation process.
"""

import logging
import os
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import tensorflow as tf

from .attention_map import generate_attention_map, generate_attention_map_with_tumor_cell
from .img_reconstruction_dto import ImageReconstructionDTO
from bcs_kedro.commons.create_tiles import slice_image_into_tiles_memory

logger = logging.getLogger(__name__)

###########################
#    Helper Functions     #
###########################


def reconstruct_ihc_image(
    ordered_image_list,
    tile_width,
    tile_height,
    orig_img_width,
    orig_img_height,
    nchannels=1,
):
    """
    Reconstructs IHC image from tiles.

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
    logger.debug("Reading IHC Image files as a flat array")
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


def pad_tile_matrices(image_tiles_contours, dim=2):
    logger.debug("Padding matrix dimensions")
    max_length_dim = max(
        [tile_contours.shape[dim] for tile_contours in image_tiles_contours]
    )
    for i, tile_contours in enumerate(image_tiles_contours):
        logger.debug("Padding tile (%d of %d)", i + 1, len(image_tiles_contours))
        padding_size = max_length_dim - tile_contours.shape[dim]
        pads = [
            (0, padding_size) if d == dim else (0, 0)
            for d in range(len(tile_contours.shape))
        ]
        image_tiles_contours[i] = np.pad(tile_contours, pads, mode="constant")
        logger.debug("Padded tile shape: %s", image_tiles_contours[i].shape)
    return image_tiles_contours


def reconstruct_seg_image(
    ordered_image_list,
    tile_width,
    tile_height,
    orig_img_width,
    orig_img_height,
):
    """Reconstruct the segmented image from tiles.

    Args:
        ordered_image_list (list): Ordered image list of tile image_paths.
        tile_width (int): Width of each tile in pixels.
        tile_height (int): Height of each tile in pixels.
        orig_img_width (int): Width of the original image.
        orig_img_height (int): Height of the original image.

    Returns:
        list[list[numpy.ndararray]]: Reconstructed segmented images.
    """
    logger.info("Extracting contours from .npz segmented images")
    contour_detections = []

    for img_tile_path in ordered_image_list:
        contours = np.load(img_tile_path)["contours"]
        if len(contours):
            contours = np.transpose(contours, (1, 2, 0))
        else:
            contours = np.zeros((tile_width, tile_height, 1))

        contour_detections.append(contours)

    num_tiles_per_row = orig_img_width // tile_width
    num_tiles_per_col = orig_img_height // tile_height

    logger.debug("Reshaping contours of segmented image")
    contour_detections = [
        contour_detections[i : i + num_tiles_per_col]
        for i in range(0, len(contour_detections), num_tiles_per_row)
    ]
    return contour_detections


def get_reconstructed_images_from_metadata_sample(
    df_image_sample, tile_width, tile_height, orig_img_width, orig_img_height
):
    """Reconstructs all the IHC and segmented images from a tile metadata sample.

    Args:
        df_image_sample (pd.DataFrame): Sample image with all the tile metadata.
        tile_width (int): The width of the tile in pixels.
        tile_height (int): The height of the tile in pixels.
        orig_img_width (int): The width of the original image.
        orig_img_height (int): The height of the original image.

    Returns:
        (Tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]): Tuple containing
            the reconstructed KI67, PHH3, and HES segmented images, respectively.
    """

    img_ki67 = reconstruct_ihc_image(
        df_image_sample.path_ki67.tolist(),
        tile_width=tile_width,
        tile_height=tile_height,
        orig_img_width=orig_img_width,
        orig_img_height=orig_img_height,
        nchannels=1,
    )

    img_phh3 = reconstruct_ihc_image(
        df_image_sample.path_phh3.tolist(),
        tile_width=tile_width,
        tile_height=tile_height,
        orig_img_width=orig_img_width,
        orig_img_height=orig_img_height,
        nchannels=1,
    )

    img_seg = reconstruct_seg_image(
        df_image_sample.path_seg.tolist(),
        tile_width=tile_width,
        tile_height=tile_height,
        orig_img_width=orig_img_width,
        orig_img_height=orig_img_height,
    )

    return img_ki67, img_phh3, img_seg


def get_att_map_image_name_from_metadata_sample(sample):
    return f"{sample.slide_type}_CASE{sample.case_id}_{sample.img_id}_attention_map.npz"


def build_consolidated_metadata_table(src_dir_deconv_tiles, src_dir_segmented_tiles):
    logger.info("Loading IHC image metadata")
    ihc_metadata_path = os.path.join(src_dir_deconv_tiles, "metadata.parquet")
    df_ihc_metadata = pd.read_parquet(ihc_metadata_path)
    logger.info("Separating IHC metadata in PHH3 and Ki67 biomarkers")
    df_phh3_metadata = df_ihc_metadata.loc[df_ihc_metadata.biomarker == "PHH3", :]
    df_ki67_metadata = df_ihc_metadata.loc[df_ihc_metadata.biomarker == "KI67", :]

    merge_on_cols = [
        "target",
        "case_id",
        "img_id",
        "tile_row_offset",
        "tile_col_offset",
    ]

    logger.info("Merging PHH3 and Ki67 metadata horizontally (join)")
    df_ihc_metadata = pd.merge(
        df_phh3_metadata,
        df_ki67_metadata,
        how="inner",
        on=merge_on_cols,
        suffixes=("_phh3", "_ki67"),
    )

    logger.info("Loading HES image metadata")
    seg_metadata_path = os.path.join(src_dir_segmented_tiles, "metadata.parquet")
    df_seg_metadata = pd.read_parquet(seg_metadata_path)

    logger.info("Mergining IHC image metadata with HES images metadata")
    df_metadata = pd.merge(
        df_ihc_metadata,
        df_seg_metadata,
        how="inner",
        on=merge_on_cols,
    )
    df_metadata = df_metadata.rename(columns={"path": "path_seg"})

    logger.info("Removing redundant and duplicated information after merge")

    pattern_cols_to_filter = ["biomarker"]  # ["orig_img_path", "biomarker"]

    filter_columns_that_contain_pattern = df_metadata.columns.str.contains(
        "|".join(pattern_cols_to_filter)
    )
    cols_to_drop = df_metadata.columns[filter_columns_that_contain_pattern].tolist()
    logger.debug("Dropping columns from consolidated metadata table: %s", cols_to_drop)
    df_metadata = df_metadata.drop(columns=cols_to_drop)
    # get the first true index of a pandas series?
    df_metadata.columns = df_metadata.columns.str.replace(
        r"^orig_img_path.*", "orig_img_path", regex=True
    )
    df_metadata = df_metadata.loc[:, ~df_metadata.columns.duplicated()].copy()

    df_metadata.columns = df_metadata.columns.str.replace(
        r"(?<!path)_(phh3|ki67|seg)", "", regex=True
    )
    logger.debug("Dropping duplicated columns")
    df_metadata = df_metadata.loc[:, ~df_metadata.columns.duplicated()]
    logger.debug("Sorting columns in alphabetical order")
    return df_metadata.reindex(sorted(df_metadata.columns), axis=1)


def ihc_and_hes_image_generator(
    src_dir_deconv_tiles,
    src_dir_segmented_tiles,
    tile_width,
    tile_height,
    orig_img_width,
    orig_img_height,
):
    """
    Generates tuples of IHC and HES image paths for all images in the dataset.

    Args:
        src_dir_deconv_tiles (str): Path to the directory containing the deconvolved tiles.
        src_dir_segmented_tiles (str): Path to the directory containing the segmented tiles.
        tile_width (int): The width of the tile in pixels.
        tile_height (int): The height of the tile in pixels.
        orig_img_width (int): The width of the original image.
        orig_img_height (int): The height of the original image.

    Returns:
        (ImageReconstructionDTO): A generator for ImageReconstruction DTO objects containing
            IHC and HES reconstructed images with additional metadata.
    """
    df_metadata = build_consolidated_metadata_table(
        src_dir_deconv_tiles, src_dir_segmented_tiles
    )

    logger.info("Iterating over the merged metadata to yield image paths")
    while not df_metadata.empty:
        # Get the first row of the dataframe
        sample = df_metadata.iloc[0]
        df_image_sample = df_metadata.query(
            "(target == @sample.target) and"
            "(case_id == @sample.case_id) and"
            "(img_id == @sample.img_id)"
        )
        df_image_sample = df_image_sample.sort_values(
            by=["tile_row_offset", "tile_col_offset"], ascending=True
        )

        img_ki67, img_phh3, img_seg = get_reconstructed_images_from_metadata_sample(
            df_image_sample, tile_width, tile_height, orig_img_width, orig_img_height
        )
        att_map_img_name = get_att_map_image_name_from_metadata_sample(sample)

        logger.info("Dropping image sample from dataframe")
        df_metadata.drop(index=df_image_sample.index, inplace=True)

        image_reconstruction_info = ImageReconstructionDTO(
            img_ki67=img_ki67,
            img_phh3=img_phh3,
            img_seg=img_seg,
            output_image_path=os.path.join(sample.target, att_map_img_name),
            metadata=df_image_sample,
        )

        yield image_reconstruction_info


def attention_map_already_exists(
    base_output_path, orig_img_width, orig_img_height, tile_width, tile_height
):
    base_output_no_ext, ext = os.path.splitext(os.path.abspath(base_output_path))
    glob_pattern = f"{base_output_no_ext}*"
    expected_attmap_num_tiles = (orig_img_height // tile_height) * (
        orig_img_width // tile_width
    )
    logger.info("Searching for glob pattern: %s", glob_pattern)
    len_tile_paths = len(glob(glob_pattern))
    return len_tile_paths == expected_attmap_num_tiles

###########################
#         Nodes           #
###########################


def generate_attention_maps(
    src_dir_deconv_tiles,
    src_dir_segmented_tiles,
    dst_dir_attention_maps,
    tile_width,
    tile_height,
    orig_img_width,
    orig_img_height,
    bandwidth,
    kde_samples,
    use_tumor_cell
):
    """
    Generates attention maps for all images in the dataset.

    Args:
        src_dir_deconv_tiles (str): Path to the directory containing the deconvolved tiles.
        src_dir_segmented_tiles (str): Path to the directory containing the segmented tiles.
        dst_dir_attention_maps (str): Path to the directory where the attention maps will be saved.
        tile_width (int): The width of the tile in pixels.
        tile_height (int): The height of the tile in pixels.
        orig_img_width (int): The width of the original image.
        orig_img_height (int): The height of the original image.
        bandwidth (int): The bandwidth of the gaussian kernels in the KDE.
        kde_samples (int): The number of samples used in the grid of KDE probabilities.
            The output size of the attention map is number of sampled KDE probabilities
            (kde_samples x kde_samples).
    """
    att_map_metadata = []
    logger.info("Generating attention maps")
    for reconstructed_imgs in ihc_and_hes_image_generator(
        src_dir_deconv_tiles,
        src_dir_segmented_tiles,
        tile_width,
        tile_height,
        orig_img_width,
        orig_img_height,
    ):  
        if attention_map_already_exists(
            base_output_path=os.path.join(
                dst_dir_attention_maps, reconstructed_imgs.output_image_path
            ),
            orig_img_width=orig_img_width,
            orig_img_height=orig_img_height,
            tile_width=tile_width,
            tile_height=tile_height,
        ):
            logger.info(
                "Attention map already exists for image: IMG ID: %s | CASE ID: %s | TARGET: %s",
                reconstructed_imgs.metadata.img_id.iloc[0],
                reconstructed_imgs.metadata.case_id.iloc[0],
                reconstructed_imgs.metadata.target.iloc[0],
            )
            logger.info("Skipping image")
            continue
        
        
        logger.info(
            "Generating attention map for image: IMG ID: %s | CASE ID: %s | TARGET: %s",
            reconstructed_imgs.metadata.img_id,
            reconstructed_imgs.metadata.case_id,
            reconstructed_imgs.metadata.target,
        )
        if(use_tumor_cell):
            img_att_map = generate_attention_map_with_tumor_cell(
                segmented_image=reconstructed_imgs.img_seg,
                deconv_img_ki67=reconstructed_imgs.img_ki67,
                deconv_img_phh3=reconstructed_imgs.img_phh3,
                bandwidth=bandwidth,
                kde_samples=kde_samples,
            )
        else:
            img_att_map = generate_attention_map(
                deconv_img_ki67=reconstructed_imgs.img_ki67,
                deconv_img_phh3=reconstructed_imgs.img_phh3,
                bandwidth=bandwidth,
                kde_samples=kde_samples,
            )
        logger.info("Attention map generated")

        logger.info("Slicing attention map in tiles")
        tile_generator = slice_image_into_tiles_memory(
            img=img_att_map,
            base_output_path=reconstructed_imgs.output_image_path,
            out_dir=dst_dir_attention_maps,
            out_file_ext=".npz",
            tile_width=tile_width,
            tile_height=tile_height,
        )

        att_map_paths = []

        for tile, tile_metadata in tile_generator:
            full_tile_img_output_path = os.path.join(
                dst_dir_attention_maps, tile_metadata["tile_img_path"]
            )

            os.makedirs(os.path.dirname(full_tile_img_output_path), exist_ok=True)

            logger.info(
                "Saving attention map tile to path: %s", full_tile_img_output_path
            )

            np.savez(full_tile_img_output_path, attmap=tile)
            att_map_paths.append(os.path.abspath(full_tile_img_output_path))

        reconstructed_imgs.metadata["att_map_path"] = att_map_paths
        att_map_metadata.append(reconstructed_imgs.metadata)

    df_metadata_att_map = pd.concat(att_map_metadata, axis=0)
    df_metadata_att_map.to_parquet(
        os.path.join(dst_dir_attention_maps, "metadata.parquet")
    )
