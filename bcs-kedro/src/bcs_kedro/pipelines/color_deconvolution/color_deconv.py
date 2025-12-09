import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hed
import numpy as np
from skimage.exposure import rescale_intensity
import skimage.color
from PIL import Image
from glob import glob
from skimage import measure
from time import time

from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd

def colordeconv(img, sh=0.31):
    imgb = skimage.color.rgb2hed(img)
    imgb = rescale_intensity(imgb, out_range=(0, 1))

    d = imgb[:, :, 2].copy()
    d[d >= sh] = 255
    d[d < sh] = 0

    return d


def denoise(img):
    copy_img = img.copy()

    # kernel = np.ones((3,3),np.uint8)
    # 3*3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    kernel = np.exp(-(x**2 + y**2))
    kernel = kernel / kernel.sum()

    copy_img = cv2.dilate(copy_img, kernel, iterations=2)
    copy_img = cv2.erode(copy_img, kernel, iterations=2)

    # copy_img = cv2.erode(copy_img,kernel,iterations = 2)
    # copy_img = cv2.dilate(copy_img,kernel,iterations = 2)

    return copy_img


def find_connected_component(img, area_threshold):

    rt_img = []
    for threshold in area_threshold:
        grid = img.copy()

        labels = measure.label(grid, connectivity=1)
        for region in measure.regionprops(labels):
            if region.area <= threshold:
                (min_row, min_col, max_row, max_col) = region.bbox
                grid[min_row:max_row, min_col:max_col] = 0

        rt_img.append(grid)

    return rt_img


def apply_color_deconv(
    img_path, dab_threshold=0.31
):
    imgrgb = Image.open(img_path).convert("RGB")

    start = time()

    imghed = colordeconv(imgrgb, sh=dab_threshold)

    print("Execute Time : %f sec" % (time() - start))

    return imghed


def parse_args():
    # Variable definitions
    parser = ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=str,
        default="../data/02_intermediate/spatial-attention-pcr",
        help="Source Image Directory with raw image separated by tiles.",
    )
    parser.add_argument(
        "--dst-dir", 
        type=str,
        default="../data/03_primary/spatial-attention-pcr-deconv",
        help="Destination Directory to save the deconvoluted tile images."
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        nargs='+', 
        default=[50, 100, 150],
        help="Area Threshold"
    )
    parser.add_argument(
        "--transparent-strength",
        type=float,
        default=2.5,
        help="Image transparency strength."
    )
    parser.add_argument(
        "--metadata-file",
        type=bool,
        action='store_true',
        help="Whether to generate and store the metadata file in the destination image directory (--dst-dir)."
    )

    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    main()

