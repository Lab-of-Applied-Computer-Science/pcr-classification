import logging
import pandas as pd


logger = logging.getLogger(__name__)

###########################
#         Nodes           #
###########################

def parse_original_images(raw_tile_metadata: pd.DataFrame):
    df_orig_image_metadata = raw_tile_metadata.groupby("orig_img_path").first().reset_index()
    
