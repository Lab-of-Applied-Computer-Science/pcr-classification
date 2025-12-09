import numpy as np
import os
from tqdm import tqdm

dir_path='/scratch/henrique.colonese/breast-cancer-segmentation/data/05_model_input/spatial-attention-pcr-attmaps-vs-tiles/pcr/'

dir_path='/scratch/willianjunior/large8k'
dir_path='/tmp/large8k'
dir_path='/tmp/large8ksmall'

rng = np.random.default_rng()

#for i in tqdm(range(20000)):
#    large_float_array = rng.random(size=(250, 250)).astype(np.float32)
#    np.save(f'/tmp/large8ksmall/8klarge{i}.npy', large_float_array)

#0/0

all_avg = 0
for file in tqdm(os.listdir(dir_path)):
    filename = os.fsdecode(file)
    #f = np.load(f'{dir_path}/{filename}')["attmap"].astype(np.float32)
    f = np.load(f'{dir_path}/{filename}').astype(np.float32)

    #print(f.shape)
    avg = f.mean()
    all_avg += avg
    #print(avg)
    #0/0

