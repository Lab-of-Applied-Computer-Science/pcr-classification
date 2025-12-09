import os
import numpy as np 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

class PCRClassifier:
    def __init__(self):
        self.lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        self.early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        
    
    #def train():
    