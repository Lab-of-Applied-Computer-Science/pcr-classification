import os
os.environ['TORCH_HOME'] = '/home_cerberus/speed/henrique.colonese/Virtual_Staining/'
os.environ['MPLCONFIGDIR'] = '/home_cerberus/speed/henrique.colonese/temp/'
import sys
import cv2
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models
import torch.nn.init as init
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import shutil
import torchvision
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torchvision.utils import save_image
from itertools import product
from tqdm import tqdm
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from pathlib import Path
import json

current_time = datetime.now()
date_str = current_time.strftime("%Y%m%d_%H%M")
print(date_str)

# stain = "KI67"
stain = "PHH3"

metrica = "SSIM"


BASE_DIR = f"/scratch/henrique.colonese/Virtual_Staining/{stain}/{metrica}/results_{date_str}"
os.makedirs(f"{BASE_DIR}", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/results", exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, img_paths, target_paths, resize=(512, 512), augment=True):
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.resize = resize
        self.augment = augment
        self.transform = PairRandomTransform(augment=augment)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load source image (x)
        img = cv2.imread(self.img_paths[idx])
        img = img[:, :, ::-1]  # BGR to RGB (if model expects RGB)
        img = np.ascontiguousarray(img)
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C,H,W]

        # Load target image (y)
        target = cv2.imread(self.target_paths[idx])
        target = target[:, :, ::-1]  # BGR to RGB (consistent with img)
        target = np.ascontiguousarray(target)
        target = cv2.resize(target, self.resize, interpolation=cv2.INTER_AREA)
        target = target.astype(np.float32) / 255.0  # Normalize
        target = torch.from_numpy(target).permute(2, 0, 1)  # [C,H,W]

        return img, target  # Both are 3x512x512 tensors

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) 
        )

    def forward(self, x):
        return self.skip(x)

class CycleGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual=4):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 32, 7),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features*2
        
        # Residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features//2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, out_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, img):
        return self.model(img)  # Outputs [batch, 1, 30, 30] for 512x512 input

def LoadData(path, parquet_path, max_black_ratio=0.3):
    metadata_df = pd.read_parquet(parquet_path)
    black_ratio_dict = {
        os.path.basename(p): ratio 
        for p, ratio in zip(metadata_df['path'], metadata_df['black_pixel_ratio'])
    }
    
    file_list = os.listdir(path)
    orig_img = []
    mask_img = []
    missing_masks = 0
    
    for file in file_list:
        if file.endswith('.jpg'):
            if 'HES' in file:
                
                
                parts = file.split('_')
                patient_id = parts[1]  # e.g., CASE42
                image_id = parts[3] + "_" + parts[4]  # e.g., 70597_45445
                tile_coordinates = '_'.join(parts[5:]).replace('.jpg', '')
                mask_name = f"rigid_{patient_id}_KI-67_{image_id}_{tile_coordinates}.jpg"
                
                if mask_name in file_list:
                    if black_ratio_dict[mask_name] < max_black_ratio:
                        orig_img.append(file)
                        mask_img.append(mask_name)
                    else:
                        missing_masks += 1
                        print(f"Mask not in metadata: {mask_name}")

    print(f"\nTotal masks missing from metadata: {missing_masks}")
    print(f"Found {len(orig_img)} valid image pairs")

    orig_img.sort()
    mask_img.sort()
    
    # Return FULL PATHS (not just filenames)
    orig_img_paths = [os.path.join(path, file) for file in orig_img]
    mask_img_paths = [os.path.join(path, file) for file in mask_img]
    
    return orig_img_paths, mask_img_paths

def get_path(image_filename):
    basename = os.path.basename(image_filename)
    
    for folder in [PCR, NPCR]:
        full_path = os.path.join(dataset_dir + folder, basename)
        if os.path.exists(full_path):
            return full_path
    print("NONE FOUND")
    return None  # Explicitly return None if not found

def LoadData_CSV(csv_path, metric, min_miou=0.6):
    df = pd.read_csv(csv_path)
    
    # First get corrected paths
    df['corrected_HES'] = df['HES_Path'].apply(get_path)
    df['corrected_KI67'] = df['PHH3_Path'].apply(get_path)
    

    if metric == "MEDIA":
        # MEDIA
        valid_pairs = df[
            ((df['Similarity'] / 255000000) > min_miou) &
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]
    
    elif metric == "SSIM":
        # SSIM
        valid_pairs = df[
            (df['SSIM']  > min_miou) &
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]
    
    elif metric == "MI":
        # MI
        min_miou = 0.05
        valid_pairs = df[
            (df['MI']  > min_miou) &
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]
    
    elif metric =="none":
        valid_pairs = df[
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]

    print(f"Number of images to process:  {len(valid_pairs['corrected_HES'])}")
    print(f"Number of masks to process: {len(valid_pairs['corrected_KI67'])}")
    
    return valid_pairs['corrected_HES'].tolist(), valid_pairs['corrected_KI67'].tolist()

def FilterData_CSV(csv_path, metric, min_miou=0.6):
    df = pd.read_csv(csv_path)
    
    if metric == "MEDIA":
        # MEDIA
        valid_pairs = df[
            ((df['Similarity'] / 255000000) > min_miou)
        ]
    
    elif metric == "SSIM":
        # SSIM
        valid_pairs = df[
            (df['SSIM']  > min_miou)
        ]
    
    elif metric == "MI":
        # MI
        min_miou = 0.05
        valid_pairs = df[
            (df['MI']  > min_miou)
        ]
    
    print(f"Number of images to process:  {len(valid_pairs['corrected_HES'])}")
    print(f"Number of masks to process: {len(valid_pairs['corrected_KI67'])}")
    
    return valid_pairs['corrected_HES'].tolist(), valid_pairs['corrected_KI67'].tolist()

def pixel_accuracy(output, target):
    """Calculate pixel accuracy between generated and real RGB images"""
    with torch.no_grad():
        # Threshold if needed (for values outside [0,1])
        output = torch.clamp(output, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Calculate per-pixel accuracy (within some tolerance)
        correct = (torch.abs(output - target) < 0.05).float()  # 5% tolerance
        accuracy = correct.mean()
    return accuracy.item()

def m_iou(pred, target, smooth=1e-6, threshold=0.5):
    """Calculate mean IoU for float-valued images (e.g., GAN outputs)"""
    with torch.no_grad():
        # Binarize predictions and targets
        pred = (pred > threshold).float()  # Convert to 0s and 1s
        target = (target > threshold).float()
        
        # Calculate intersection and union per channel
        ious = []
        for c in range(pred.shape[1]):  # Loop through channels (R, G, B)
            intersection = (pred[:, c] * target[:, c]).sum((1, 2))  # Sum over H and W
            union = (pred[:, c] + target[:, c]).sum((1, 2)) - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou)
        
        return torch.mean(torch.stack(ious)).item()  # Average across channels

def gradient_penalty(discriminator, real, fake, device):
    """Ultra-stable WGAN-GP implementation"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1-alpha) * fake).requires_grad_(True)
    
    with torch.autocast(device_type='cuda', enabled=False):  # Disable AMP for GP
        d_interpolates = discriminator(interpolates.float())  # Force float32
        
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty.clamp(max=10.0)

# data aug
class PairRandomTransform:
    """Apply the same random transformation to both image and mask."""
    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, img, mask):
        if self.augment:
            # Random horizontal flip (same for both)
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            # Random vertical flip (same for both)
            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)


        # Convert to tensor and normalize
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        return img, mask

def split_by_patient(img_paths, target_paths, test_size=0.1, val_size=0.15, random_state=42):

    # Step 1: Extract unique patient IDs (e.g., "CASE10")
    def extract_patient_id(path):
        filename = os.path.basename(path)
        return filename.split('_')[1]
    
    patient_ids = list(set(extract_patient_id(path) for path in img_paths))
    print(f"Total unique patients: {len(patient_ids)}")
    
    # Step 2: Split patient IDs into train/val/test
    patients_train, patients_test = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    patients_train, patients_val = train_test_split(
        patients_train, test_size=val_size / (1 - test_size), random_state=random_state
    )
    
    # Step 3: Assign images to splits based on patient IDs
    def filter_by_patients(patients, img_paths, target_paths):
        img_selected = []
        target_selected = []
        for img_path, target_path in zip(img_paths, target_paths):
            if extract_patient_id(img_path) in patients:
                img_selected.append(img_path)
                target_selected.append(target_path)
        return img_selected, target_selected
    
    img_train, target_train = filter_by_patients(patients_train, img_paths, target_paths)
    img_val, target_val = filter_by_patients(patients_val, img_paths, target_paths)
    img_test, target_test = filter_by_patients(patients_test, img_paths, target_paths)
    
    # Log statistics
    print(f"Train patients: {len(patients_train)} ({len(img_train)} images)")
    print(f"Val patients: {len(patients_val)} ({len(img_val)} images)")
    print(f"Test patients: {len(patients_test)} ({len(img_test)} images)")
    
    return img_train, img_val, img_test, target_train, target_val, target_test

def save_splits(splits, filename="dataset_splits_filter.json"):
    """Save dataset splits to a JSON file"""
    split_dict = {
        'img_train': splits[0],
        'img_val': splits[1],
        'img_test': splits[2],
        'target_train': splits[3],
        'target_val': splits[4],
        'target_test': splits[5]
    }
    
    with open(filename, 'w') as f:
        json.dump(split_dict, f, indent=2)
        
    print(f"Splits saved to {filename}")
    
def load_split_json(filename="dataset_splits_filter.json"):
    """Load dataset splits from a JSON file"""
    with open(filename, 'r') as f:
        split_dict = json.load(f)
        
    return (
        split_dict['img_train'],
        split_dict['img_val'],
        split_dict['img_test'],
        split_dict['target_train'],
        split_dict['target_val'],
        split_dict['target_test']
    )

def check_leakage(img_train, img_val, img_test):
    
    def extract_patient_id(path):
        filename = os.path.basename(path)
        return filename.split('_')[1]
    
    def get_patients(paths):
        return set(extract_patient_id(p) for p in paths)
    
    train_patients = get_patients(img_train)
    val_patients = get_patients(img_val)
    test_patients = get_patients(img_test)
    
    assert train_patients.isdisjoint(val_patients), "Train/Val leakage!"
    assert train_patients.isdisjoint(test_patients), "Train/Test leakage!"
    print("✅ No patient leakage detected.")

def load_splits(csv_path):
    df = pd.read_csv(csv_path)
    
    img_train = df[df['split'] == 'train']['image_path'].tolist()
    target_train = df[df['split'] == 'train']['target_path'].tolist()
    
    img_valid = df[df['split'] == 'valid']['image_path'].tolist()
    target_valid = df[df['split'] == 'valid']['target_path'].tolist()
    
    img_test = df[df['split'] == 'test']['image_path'].tolist()
    target_test = df[df['split'] == 'test']['target_path'].tolist()
    
    return img_train, img_valid, img_test, target_train, target_valid, target_test

# obtendo diretorios
dataset_dir = '../../daniel.ayala/breast-cancer-segmentation/bcs-kedro/data/02_intermediate/spatial-attention-pcr/'
PCR = '/pcr'
NPCR = '/non-pcr'
parquet_path = './metadata_with_bpr.parquet'

num_classes = 2
batch_size = 2
target_shape = (512, 512)
path = dataset_dir

print("Starting...")
print("Loading file paths...")

# img_paths, target_paths = LoadData(path, parquet_path)
img_paths, target_paths = LoadData_CSV(
    csv_path='./PHH3_Similarity_Full_Dataset.csv',
    metric="none",
    min_miou=0.6
)

full_df = pd.read_csv('./PHH3_Similarity_Full_Dataset.csv')
full_df['corrected_HES'] = full_df['HES_Path'].apply(get_path)
full_df['corrected_KI67'] = full_df['PHH3_Path'].apply(get_path)

print("Splitting dataset paths...")

img_train, img_valid, img_test, target_train, target_valid, target_test = load_split_json()

metrica_dict = metrica
if metrica == "MEDIA":
    metrica_dict = "Similarity"


dict_lookup = {
    (row['corrected_HES'], row['corrected_KI67']): row[metrica_dict] 
    for _, row in full_df.iterrows()
}

train_metric_values = [dict_lookup[(img, target)] for img, target in zip(img_train, target_train)]
metric_train_threshold = np.quantile(train_metric_values, 0.80)

valid_metric_values = [dict_lookup[(img, target)] for img, target in zip(img_valid, target_valid)]
metric_valid_threshold = np.quantile(valid_metric_values, 0.80)

test_metric_values = [dict_lookup[(img, target)] for img, target in zip(img_test, target_test)]
metric_test_threshold = np.quantile(test_metric_values, 0.80)

print(f"Threshold {metrica} usado em train: {metric_train_threshold:.3f}")
print(f"Threshold {metrica} usado em val: {metric_valid_threshold:.3f}")
print(f"Threshold {metrica} usado em test: {metric_test_threshold:.3f}")

split_data = {
    'split': [],
    'image_path': [],
    'target_path': []
}

for img, target in zip(img_train, target_train):
    if dict_lookup.get((img, target), 0) >= metric_train_threshold:
        split_data['split'].append('train')
        split_data['image_path'].append(img)
        split_data['target_path'].append(target)
        
for img, target in zip(img_valid, target_valid):
    if dict_lookup.get((img, target), 0) >= metric_valid_threshold:
        split_data['split'].append('valid')
        split_data['image_path'].append(img)
        split_data['target_path'].append(target)

for img, target in zip(img_test, target_test):
    if dict_lookup.get((img, target), 0) >= metric_test_threshold:
        split_data['split'].append('test')
        split_data['image_path'].append(img)
        split_data['target_path'].append(target)


df_splits = pd.DataFrame(split_data)
df_splits.to_csv(f"./dataset_splits_{stain}_{metrica}.csv", index=False)
print("Splits salvos com sucesso!")

img_train, img_valid, img_test, target_train, target_valid, target_test = load_splits(f"./dataset_splits_{stain}_{metrica}.csv")

check_leakage(img_train, img_valid, img_test)

print(f"Training samples: {len(img_train)}")
print(f"Validation samples: {len(img_valid)}")
print(f"Test samples: {len(img_test)}")

print("Creating datasets...")
train_dataset = CustomDataset(img_train, target_train, resize=target_shape, augment=True)
print("Train created")
valid_dataset = CustomDataset(img_valid, target_valid, resize=target_shape, augment=False)
print("Val created")
test_dataset = CustomDataset(img_test, target_test, resize=target_shape, augment=False)
print("Test created")

num_workers = 4
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)
print("Dataloaders created")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 150
patience = 15
trigger_times = 0
best_miou = -float('inf')
best_pixel_acc = 0.0


G_he2ki67 = CycleGenerator().to(device)  # H&E to KI-67
G_ki672he = CycleGenerator().to(device)  # KI-67 to H&E
D_he = PatchGANDiscriminator().to(device)  # Discriminator for H&E
D_ki67 = PatchGANDiscriminator().to(device)  # Discriminator for KI-67
print("Models initialized")

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_ssim = SSIM(
    data_range=1.0,
    kernel_size=11,
    reduction='elementwise_mean'
).to(device)

# Optimizers
optimizer_G = optim.Adam(
    list(G_he2ki67.parameters()) + list(G_ki672he.parameters()),
    lr=0.0004, betas=(0.5, 0.999)
)
optimizer_D = optim.Adam(
    list(D_he.parameters()) + list(D_ki67.parameters()),
    lr=0.0004, betas=(0.5, 0.999)
)

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

print("Entering training loop")

scaler_G = GradScaler()
scaler_D = GradScaler()

# Training loop
for epoch in range(num_epochs):
    G_he2ki67.train()
    G_ki672he.train()
    D_he.train()
    D_ki67.train()

    train_g_loss, train_d_loss = 0.0, 0.0
    train_pix_acc, train_miou = 0.0, 0.0
    
    for he_images, ki67_images in train_loader:
        he_images = he_images.to(device)
        ki67_images = ki67_images.to(device)
        batch_size = he_images.size(0)
        
        optimizer_G.zero_grad()
        
        with autocast():
            fake_ki67 = G_he2ki67(he_images)
            fake_he = G_ki672he(ki67_images)
            

            valid = torch.ones_like(D_ki67(fake_ki67))
            

            loss_id_he = criterion_identity(G_ki672he(he_images), he_images)
            loss_id_ki67 = criterion_identity(G_he2ki67(ki67_images), ki67_images)
            loss_identity = (loss_id_he + loss_id_ki67) / 2
            
            loss_GAN_he2ki67 = criterion_GAN(D_ki67(fake_ki67), valid)
            loss_GAN_ki672he = criterion_GAN(D_he(fake_he), valid)
            
            reconstructed_he = G_ki672he(fake_ki67)
            loss_cycle_he = criterion_cycle(reconstructed_he, he_images)
            
            reconstructed_ki67 = G_he2ki67(fake_he)
            loss_cycle_ki67 = criterion_cycle(reconstructed_ki67, ki67_images)
            loss_cycle = (loss_cycle_he + loss_cycle_ki67) / 2
            
            loss_ssim_ki67 = 1 - criterion_ssim(fake_ki67, ki67_images)
            loss_ssim_he = 1 - criterion_ssim(fake_he, he_images)
            loss_ssim = (loss_ssim_ki67 + loss_ssim_he) / 2

            loss_G = (
                2.0 * loss_GAN_he2ki67 + 
                2.0 * loss_GAN_ki672he +
                3.0 * loss_cycle +  
                1.0 * loss_identity +  
                0.5 * loss_ssim
            )

        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        optimizer_D.zero_grad()
        

        with autocast():
            with torch.no_grad():
                fake_ki67 = G_he2ki67(he_images)
                fake_he = G_ki672he(ki67_images)
            
            valid = torch.ones_like(D_ki67(ki67_images))
            fake = torch.zeros_like(D_ki67(fake_ki67))
            
            loss_real_he = criterion_GAN(D_he(he_images), valid)
            loss_real_ki67 = criterion_GAN(D_ki67(ki67_images), valid)
            loss_fake_he = criterion_GAN(D_he(fake_he.detach()), fake)
            loss_fake_ki67 = criterion_GAN(D_ki67(fake_ki67.detach()), fake)
            
            lambda_gp = 0.001
            gp_he = gradient_penalty(D_he, he_images, fake_he.detach(), device)
            gp_ki67 = gradient_penalty(D_ki67, ki67_images, fake_ki67.detach(), device)
            
            loss_D = (
                (loss_real_he + loss_fake_he + loss_real_ki67 + loss_fake_ki67) / 4 + 
                lambda_gp * (gp_he + gp_ki67) / batch_size
            )
        

        scaler_D.scale(loss_D).backward()
        
        torch.nn.utils.clip_grad_norm_(D_he.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(D_ki67.parameters(), 0.1)
        
        scaler_D.step(optimizer_D)
        scaler_D.update()

        with torch.no_grad():
            fake_ki67 = torch.clamp(fake_ki67, 0, 1)
            batch_pix_acc = pixel_accuracy(fake_ki67, ki67_images)
            batch_miou = m_iou(fake_ki67, ki67_images)
        
        train_g_loss += loss_G.item() * batch_size
        train_d_loss += loss_D.item() * batch_size
        train_pix_acc += batch_pix_acc * batch_size
        train_miou += batch_miou * batch_size
    
    train_g_loss /= len(train_loader.dataset)
    train_d_loss /= len(train_loader.dataset)
    train_pix_acc /= len(train_loader.dataset)
    train_miou /= len(train_loader.dataset)

    G_he2ki67.eval()
    D_ki67.eval()
    valid_g_loss, valid_d_loss, valid_pix_acc, valid_miou = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad(), autocast():
        for he_images, ki67_images in valid_loader:
            he_images = he_images.to(device)
            ki67_images = ki67_images.to(device)
            batch_size = he_images.size(0)
            
            fake_ki67 = G_he2ki67(he_images)
            fake_ki67 = torch.clamp(fake_ki67, 0, 1)
            
            d_real = D_ki67(ki67_images)
            d_fake = D_ki67(fake_ki67)
            valid = torch.ones_like(d_real)
            fake = torch.zeros_like(d_fake)
            
            real_loss = criterion_GAN(d_real, valid)
            fake_loss = criterion_GAN(d_fake, fake)
            valid_d_loss += (real_loss + fake_loss).item() / 2 * batch_size
            valid_g_loss += criterion_GAN(d_fake, valid).item() * batch_size
            
            valid_pix_acc += pixel_accuracy(fake_ki67, ki67_images) * batch_size
            valid_miou += m_iou(fake_ki67, ki67_images) * batch_size

    valid_g_loss /= len(valid_loader.dataset)
    valid_d_loss /= len(valid_loader.dataset)
    valid_pix_acc /= len(valid_loader.dataset)
    valid_miou /= len(valid_loader.dataset)
    
    scheduler_G.step(valid_g_loss)
    scheduler_D.step(valid_d_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}, "
          f"Pix Acc: {train_pix_acc:.4f}, mIoU: {train_miou:.4f} | "
          f"Valid G Loss: {valid_g_loss:.4f}, "
          f"Pix Acc: {valid_pix_acc:.4f}, mIoU: {valid_miou:.4f}")
    
    if valid_miou > best_miou or (valid_miou == best_miou and valid_pix_acc > best_pixel_acc):
        best_miou = valid_miou
        best_pixel_acc = valid_pix_acc
        torch.save({
            'epoch': epoch,
            'G_he2ki67_state_dict': G_he2ki67.state_dict(),
            'G_ki672he_state_dict': G_ki672he.state_dict(),
            'D_he_state_dict': D_he.state_dict(),
            'D_ki67_state_dict': D_ki67.state_dict(),
            'best_miou': best_miou,
            'best_pixel_acc': best_pixel_acc
        }, f"{BASE_DIR}/models/best_cyclegan_model_{epoch+1}.pth")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

G_he2ki67.eval()
test_loss, test_pix_acc, test_miou = 0.0, 0.0, 0.0
generated_ki67 = []
real_ki67 = []
he_images = []

virtualization_dir = os.path.join(BASE_DIR, "virtualized_results")
os.makedirs(virtualization_dir, exist_ok=True)

with torch.no_grad():
    for idx, (he_imgs, ki67_imgs) in enumerate(test_loader):
        he_imgs = he_imgs.to(device)
        ki67_imgs = ki67_imgs.to(device)
        
        fake_ki67 = G_he2ki67(he_imgs)
        fake_ki67 = torch.clamp(fake_ki67, 0, 1)
        
        he_images.append(he_imgs.cpu())
        generated_ki67.append(fake_ki67.cpu())
        real_ki67.append(ki67_imgs.cpu())
        
        test_loss += criterion_cycle(fake_ki67, ki67_imgs).item() * he_imgs.size(0)
        test_pix_acc += pixel_accuracy(fake_ki67, ki67_imgs) * he_imgs.size(0)
        test_miou += m_iou(fake_ki67, ki67_imgs) * he_imgs.size(0)
        
        for i in range(he_imgs.size(0)):
            dataset_idx = idx * test_loader.batch_size + i
            orig_path = test_dataset.img_paths[dataset_idx]
            file_name = f"virtualized_{Path(orig_path).name}"
            
            save_image(
                fake_ki67[i].cpu(),
                os.path.join(virtualization_dir, file_name),
                normalize=True
            )
        
        if idx % 10 == 0:
            for i in range(min(2, he_imgs.size(0))):
                triplet = torch.cat([
                    he_imgs[i].cpu(),
                    fake_ki67[i].cpu(),
                    ki67_imgs[i].cpu()
                ], dim=-1)
                save_image(
                    triplet,
                    os.path.join(f"{BASE_DIR}/results", f'triplet_{idx*batch_size+i}.png'),
                    normalize=True
                )

test_loss /= len(test_loader.dataset)
test_pix_acc /= len(test_loader.dataset)
test_miou /= len(test_loader.dataset)

print("\n=== Final Test Results ===")
print(f"Pixel Accuracy: {test_pix_acc:.4f}")
print(f"mIoU: {test_miou:.4f}")
print(f"Cycle Consistency Loss: {test_loss:.4f}")


print(f"Modelo treinado com {num_classes} classes, batch de tamanho {batch_size}\n")
print("Testing complete.")