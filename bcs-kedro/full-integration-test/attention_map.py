import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import sparse
import cv2
from sklearn.neighbors import KernelDensity


def generate_attention_map(
        segmented_image: np.matrix, 
        deconv_img_ki67: Image,
        deconv_img_phh3: Image,
        tile_size=(1000, 1000),
        orig_img_size=(8000,8000),
        kernel_size=(100, 100)):
    
    finalmask = np.zeros(orig_img_size)
    counts = 0
    cells_x = []
    cells_y = []
    cellski_x = []
    cellski_y = []
    cellsphh_x = [] 
    cellsphh_y = []

    kide = np.array(deconv_img_ki67)
    kide = kide[:, :,0].reshape(orig_img_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    kide = cv2.dilate(kide, kernel, iterations=2)
    kide[kide > 0] = 1

    phhde = np.array(deconv_img_phh3)
    phhde = phhde[:, :,0].reshape(orig_img_size)
    phhde = cv2.dilate(phhde, kernel, iterations=2)
    phhde[phhde > 0] = 1

    num_tiles_x = orig_img_size[0] // tile_size[0]
    num_tiles_y = orig_img_size[1] // tile_size[1]

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            masks = segmented_image[i, j]
            masks = masks.todense()
            if masks.size == 0:
                continue
            
            subki = kide[i * tile_size[0] : i * tile_size[0] + tile_size[0], j * tile_size[1] : j * tile_size[1] + tile_size[1]]
            subphh = phhde[i * tile_size[0] : i * tile_size[0] + tile_size[0], j * tile_size[1] : j * tile_size[1] + tile_size[1]]

            counts += masks.shape[-1]
            # finalmask[i*1000:i*1000+1000, j*1000:j*1000+1000]=np.sum(masks,axis=-1)
            xf = i * tile_size[0]
            yf = j * tile_size[1]

            for k in range(masks.shape[-1]):
                mask = masks[:, :, k]
                x, y = np.where(mask == 1)
                cells_x.append(np.mean(x) + xf)
                cells_y.append(np.mean(y) + yf)
                if subki[int(cells_x[-1]), int(cells_y[-1])] > 0:
                    cellski_x.append(np.mean(x) + xf)
                    cellski_y.append(np.mean(y) + yf)
                if subphh[int(cells_x[-1]), int(cells_y[-1])] > 0:
                    cellsphh_x.append(np.mean(x) + xf)
                    cellsphh_y.append(np.mean(y) + yf)

    if not cellski_y + cellsphh_y:
        return finalmask[0:orig_img_size[0] // kernel_size[0]]
    xy_train = np.vstack([cellski_y + cellsphh_y, cellski_x + cellsphh_x]).T
    step_x, step_y = orig_img_size[0] // kernel_size[0], orig_img_size[1] // kernel_size[1]
    xx, yy = np.mgrid[0:orig_img_size[0]:step_x, 0:orig_img_size[1]:step_y]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

    kde_skl = KernelDensity(bandwidth=150)
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))
    zz = np.reshape(z, xx.shape)
    return cv2.resize(zz, orig_img_size)


if __name__ == "__main__":
    filename = "55_31477_47870"

    zz = generate_attention_map(filename)
    zz = cv2.resize(zz, (8000, 8000))

    np.asve("./attenMap/{}.npy".format(filename), zz)
