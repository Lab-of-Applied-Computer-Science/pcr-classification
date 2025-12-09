import logging
from PIL import Image
import os
import pandas as pd
from .color_deconv import apply_color_deconv

logger = logging.getLogger(__name__)

###########################
#    Helper Functions     #
###########################


def _ihc_img_generator(src_dir):
    """Generates the IHC images metadata stored in the source directory.

    Args:
        src_dir (str): Path to the source directory containing the IHC images.	
    """
    logger.info('Starting IHC image generator')
    df_metadata = pd.read_parquet(os.path.join(src_dir, 'metadata.parquet'))
    df_metadata.query('biomarker == "PHH3" or biomarker == "KI67"', inplace=True)
    for _, row in df_metadata.iterrows():
        yield row
    logger.info('Finished IHC image generator')


###########################
#         Nodes           #
###########################

def run_color_deconv(src_dir, dst_dir, dab_threshold):
    """Runs the color deconvolution operation in all IHC images (PHH3 and Ki67) in the source directory.

    The deconvoluted images are saved in the destinantion directory (dst_dir) following the same folder hierarchy.

    Args:
        src_dir (str): Path to the source directory containing the IHC images.
        dst_dir (str): Path to the destination directory where the deconvoluted images will be saved.
        dab_threshold (float): Threshold value for the DAB channel.
    """

    # Create the destination directory if it doesn't exist
    logger.info('Creating destination image directory')
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over the IHC images in the source directory
    deconv_metadata = []
    for ihc_image_metadata in _ihc_img_generator(src_dir):
        logger.info('Processing image: %s', ihc_image_metadata.path)
        image_path = ihc_image_metadata.path
        image_name = os.path.basename(image_path)

        # Apply color deconvolution to the IHC image
        logger.debug('Applying color deconvolution to image')
        deconvoluted_image = apply_color_deconv(image_path, dab_threshold)
        
        # Save the deconvoluted image in the destination directory
        dst_folder_path = os.path.join(dst_dir, ihc_image_metadata.target) 
        dst_image_path = os.path.join(dst_folder_path, image_name)
        os.makedirs(dst_folder_path, exist_ok=True)
        logger.debug('Saving deconvoluted image: %s', dst_image_path)
        pil_image = Image.fromarray(deconvoluted_image).convert('RGB')
        pil_image.save(dst_image_path)

        ihc_image_metadata['path'] = dst_image_path
        deconv_metadata.append(ihc_image_metadata)

    logger.info('Saving metadata file')
    df_deconv_metadata = pd.DataFrame(deconv_metadata)
    df_deconv_metadata.to_parquet(os.path.join(dst_dir, 'metadata.parquet'))

