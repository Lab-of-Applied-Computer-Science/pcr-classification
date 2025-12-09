#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os
import numpy as np
import cv2
from itertools import product
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def get_tiles(img: np.array, tile_width: int , tile_height: int):
    img_width, img_height = img.shape[0], img.shape[1]
    tiles_offsets = product(
        range(0, img_width, tile_width), 
        range(0, img_height, tile_height))
    for col_offset, row_offset in tiles_offsets:
        tile_window = img[col_offset:(tile_width + col_offset), 
                          row_offset:(tile_height + row_offset)]
        yield col_offset, row_offset, tile_window


# In[3]:


def slice_image_into_tiles(image_path, out_dir, tile_width, tile_height):
    os.makedirs(out_dir, exist_ok=True)
    tile_metadata = []
    img = cv2.imread(image_path)
    
    for col_offset, row_offset, tile in get_tiles(img, tile_width, tile_height):
        file_name_with_ext = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(file_name_with_ext)
        tile_file_name = f"{file_name}_tile-{tile_width}x{tile_height}_x0-{col_offset}_y0-{row_offset}"
        out_path = os.path.join(
            out_dir,
            f"{tile_file_name}{file_ext}"
        )
        # if os.path.exists(out_path):
        #     print(f"Skipping tile: {out_path}")
        #     continue

        tile_metadata.append(
            {
                "tile_file_name": tile_file_name,
                "tile_img_path": out_path,
                "orig_img_width": img.shape[0],
                "orig_img_height": img.shape[1]
            }
        )
        print(f"Writing image to path: {out_path}")
        cv2.imwrite(out_path, tile)
    return tile_metadata


# In[4]:


def slice_images_into_tiles(
    root_img_dir, 
    tiles_out_dir, 
    tile_width, 
    tile_height,
    image_ext=".jpg"
):
    files_metadata = []
    for root, dirs, files in os.walk(root_img_dir):
        print(f"Finding {image_ext} image files in dir: {root}")
        for file in files:
            if file.endswith(image_ext):
                print(f"Processing image: {file}")
                path_obj = Path(os.path.join(root, file))
                file_metadata = {
                    'img_name': path_obj.parts[-1],
                    'img_path': os.path.abspath(path_obj),
                    'img_id': path_obj.parts[-2],
                    'case_id': int(path_obj.parts[-3][4:]),
                    'slide_type': path_obj.parts[-4],
                    'target_pcr': path_obj.parts[-5]
                }
                
                tile_output_target_dir = os.path.join(
                    tiles_out_dir, 
                    Path(root_img_dir).parts[-1], # Dataset root dir
                    file_metadata['target_pcr']
                )

                tiles_metadata = slice_image_into_tiles(
                    path_obj, 
                    tile_output_target_dir, 
                    tile_width, 
                    tile_height
                )
                file_metadata["tiles_metadata"] = tiles_metadata
                files_metadata.append(files_metadata)
    return files_metadata


# In[5]:


raw_img_root_dir = '../data/01_raw/spatial-attention-pcr'
image_tiles_out_dir = '../data/02_intermediate'


# In[ ]:


files_metadata = slice_images_into_tiles(
    root_img_dir=raw_img_root_dir,
    tiles_out_dir=image_tiles_out_dir,
    tile_width=1000,
    tile_height=1000
)


# In[6]:


df_metadata = pd.DataFrame(files_metadata)


# In[ ]:


df_metadata.to_csv(os.path.join(image_tiles_out_dir, 'spatial-attention-pcr', 'metadata.csv'))


# In[ ]:




