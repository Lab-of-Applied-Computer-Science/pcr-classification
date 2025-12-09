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
import torch.nn.functional as F_torch
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
        img = cv2.imread(self.img_paths[idx])
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = cv2.imread(self.target_paths[idx])
        target = target[:, :, ::-1]
        target = np.ascontiguousarray(target)
        target = cv2.resize(target, self.resize, interpolation=cv2.INTER_AREA)
        target = target.astype(np.float32) / 255.0
        target = torch.from_numpy(target).permute(2, 0, 1)

        return img, target

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
                patient_id = parts[1]
                image_id = parts[3] + "_" + parts[4]
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
    return None 

def LoadData_CSV(csv_path, metric, min_miou=0.6):
    df = pd.read_csv(csv_path)
    
    df['corrected_HES'] = df['HES_Path'].apply(get_path)
    df['corrected_KI67'] = df['PHH3_Path'].apply(get_path)
    

    if metric == "MEDIA":
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

        valid_pairs = df[
            ((df['Similarity'] / 255000000) > min_miou)
        ]
    
    elif metric == "SSIM":

        valid_pairs = df[
            (df['SSIM']  > min_miou)
        ]
    
    elif metric == "MI":

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

        output = torch.clamp(output, 0, 1)
        target = torch.clamp(target, 0, 1)
        

        correct = (torch.abs(output - target) < 0.05).float()
        accuracy = correct.mean()
    return accuracy.item()

def m_iou(pred, target, smooth=1e-6, threshold=0.5):
    """Calculate mean IoU for float-valued images (e.g., GAN outputs)"""
    with torch.no_grad():

        pred = (pred > threshold).float()
        target = (target > threshold).float()
        

        ious = []
        for c in range(pred.shape[1]):
            intersection = (pred[:, c] * target[:, c]).sum((1, 2))
            union = (pred[:, c] + target[:, c]).sum((1, 2)) - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou)
        
        return torch.mean(torch.stack(ious)).item()

def gradient_penalty(discriminator, real, fake, device):
    """Ultra-stable WGAN-GP implementation"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1-alpha) * fake).requires_grad_(True)
    
    with torch.autocast(device_type='cuda', enabled=False):
        d_interpolates = discriminator(interpolates.float())
        
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

class PairRandomTransform:
    """Apply the same random transformation to both image and mask."""
    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, img, mask):
        if self.augment:
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)


        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        return img, mask

def split_by_patient(img_paths, target_paths, test_size=0.1, val_size=0.15, random_state=42):

    def extract_patient_id(path):
        filename = os.path.basename(path)
        return filename.split('_')[1]
    
    patient_ids = list(set(extract_patient_id(path) for path in img_paths))
    print(f"Total unique patients: {len(patient_ids)}")
    

    patients_train, patients_test = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    patients_train, patients_val = train_test_split(
        patients_train, test_size=val_size / (1 - test_size), random_state=random_state
    )
    
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

class ConditionalGenerator(nn.Module):
    def __init__(self, input_channels=3, condition_channels=3, base_channels=64):
        super(ConditionalGenerator, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels + condition_channels, base_channels, 4, 2, 1),  # 512->256
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 256->128
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 128->64
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),  # 32->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, 2, 1),  # 16->32
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 4, 4, 2, 1),  # 32->64
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1),  # 64->128
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),  # 128->256
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, input_channels, 4, 2, 1),  # 256->512
            nn.Tanh()
        )
        
    def forward(self, noise, condition):
        if noise.shape[2:] != (512, 512):
            noise = F_torch.interpolate(noise, size=(512, 512), mode='bilinear', align_corners=False)
        if condition.shape[2:] != (512, 512):
            condition = F_torch.interpolate(condition, size=(512, 512), mode='bilinear', align_corners=False)
            
        x = torch.cat([noise, condition], dim=1)
        
        # Encoder
        e1 = self.enc1(x)    # (B, 64, 256, 256)
        e2 = self.enc2(e1)   # (B, 128, 128, 128)
        e3 = self.enc3(e2)   # (B, 256, 64, 64)
        e4 = self.enc4(e3)   # (B, 512, 32, 32)
        
        # Bottleneck
        b = self.bottleneck(e4)  # (B, 512, 32, 32)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))      # (B, 256, 64, 64)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))     # (B, 128, 128, 128)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))     # (B, 64, 256, 256)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))     # (B, 3, 512, 512)
        
        return d1

class ConditionalDiscriminator(nn.Module):
    def __init__(self, image_channels=3, condition_channels=3, base_channels=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(image_channels + condition_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x256 -> 128x128
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 1x1
            nn.Conv2d(base_channels * 8, 1, 16, 1, 0),
            # nn.Sigmoid()
        )
    
    def forward(self, image, condition):

        if image.shape[2:] != (512, 512):
            image = F_torch.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        if condition.shape[2:] != (512, 512):
            condition = F_torch.interpolate(condition, size=(512, 512), mode='bilinear', align_corners=False)
            

        x = torch.cat([image, condition], dim=1)
        return self.main(x).view(-1)

def generator_loss(fake_output, real_images, fake_images, lambda_l1=100):

    adversarial_loss = nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))

    l1_loss = nn.L1Loss()(fake_images, real_images)
    return adversarial_loss + lambda_l1 * l1_loss

def discriminator_loss(real_output, fake_output):

    real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2

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


img_paths, target_paths = LoadData_CSV(
    csv_path='./PHH3_Similarity_Full_Dataset.csv',
    metric="none",
    min_miou=0.6
)

full_df = pd.read_csv('./PHH3_Similarity_Full_Dataset.csv')
full_df['corrected_HES'] = full_df['HES_Path'].apply(get_path)
full_df['corrected_KI67'] = full_df['PHH3_Path'].apply(get_path)


print("Splitting dataset paths...")

img_train, img_valid, img_test, target_train, target_valid, target_test = split_by_patient(
    img_paths, 
    target_paths, 
    test_size=0.1, 
    val_size=0.15, 
    random_state=42
)

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
    # if dict_lookup.get((img, target), 0) >= 0:
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


generator = ConditionalGenerator(input_channels=3, condition_channels=3, base_channels=64).to(device)
discriminator = ConditionalDiscriminator(image_channels=3, condition_channels=3, base_channels=64).to(device)


print("Diffusion model initialized")

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, kernel_size=11, reduction='elementwise_mean').to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)

print("Entering conditional GAN training loop")
scaler_G = GradScaler()
scaler_D = GradScaler()

# Training loop
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    train_loss_G, train_loss_D = 0.0, 0.0
    train_pix_acc, train_miou = 0.0, 0.0
    
    for he_images, ki67_images in train_loader:
        he_images = he_images.to(device)
        ki67_images = ki67_images.to(device)
        batch_size = he_images.size(0)
        
        noise = torch.randn(batch_size, 3, target_shape[0], target_shape[1], device=device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        with autocast():
            fake_ki67 = generator(noise, he_images)
            fake_ki67_clamped = torch.clamp(fake_ki67, -1, 1)
            
            real_output = discriminator(ki67_images, he_images)
            fake_output = discriminator(fake_ki67_clamped.detach(), he_images)
            
            d_loss = discriminator_loss(real_output, fake_output)
        
        scaler_D.scale(d_loss).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()
        
        optimizer_G.zero_grad()
        with autocast():
            fake_output = discriminator(fake_ki67_clamped, he_images)
            fake_ki67_01 = (fake_ki67_clamped + 1) / 2
            ki67_01 = (ki67_images + 1) / 2 if ki67_images.min() < 0 else ki67_images
            g_loss = generator_loss(fake_output, ki67_01, fake_ki67_01)
        
        scaler_G.scale(g_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()
        
        with torch.no_grad():
            fake_ki67_metric = torch.clamp((fake_ki67_clamped + 1) / 2, 0, 1)
            ki67_metric = torch.clamp((ki67_images + 1) / 2, 0, 1) if ki67_images.min() < 0 else ki67_images
            batch_pix_acc = pixel_accuracy(fake_ki67_metric, ki67_metric)
            batch_miou = m_iou(fake_ki67_metric, ki67_metric)
            train_pix_acc += batch_pix_acc * batch_size
            train_miou += batch_miou * batch_size
        
        train_loss_G += g_loss.item() * batch_size
        train_loss_D += d_loss.item() * batch_size
    
    train_loss_G /= len(train_loader.dataset)
    train_loss_D /= len(train_loader.dataset)
    train_pix_acc /= len(train_loader.dataset)
    train_miou /= len(train_loader.dataset)
    
    generator.eval()
    discriminator.eval()
    valid_loss, valid_pix_acc, valid_miou = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for he_images, ki67_images in valid_loader:
            he_images = he_images.to(device)
            ki67_images = ki67_images.to(device)
            batch_size = he_images.size(0)
            

            noise = torch.randn(batch_size, 3, target_shape[0], target_shape[1], device=device)
            generated_ki67 = generator(noise, he_images)
            generated_ki67_clamped = torch.clamp(generated_ki67, -1, 1)
            generated_ki67_01 = torch.clamp((generated_ki67_clamped + 1) / 2, 0, 1)
            
            ki67_01 = torch.clamp((ki67_images + 1) / 2, 0, 1) if ki67_images.min() < 0 else ki67_images
            
            valid_loss += criterion_l1(generated_ki67_01, ki67_01).item() * batch_size
            valid_pix_acc += pixel_accuracy(generated_ki67_01, ki67_01) * batch_size
            valid_miou += m_iou(generated_ki67_01, ki67_01) * batch_size
    
    valid_loss /= len(valid_loader.dataset)
    valid_pix_acc /= len(valid_loader.dataset)
    valid_miou /= len(valid_loader.dataset)
    
    scheduler.step(valid_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: G_Loss: {train_loss_G:.4f}, D_Loss: {train_loss_D:.4f}, "
          f"Train Pix Acc: {train_pix_acc:.4f}, Train mIoU: {train_miou:.4f} | "
          f"Valid Loss: {valid_loss:.4f}, Valid Pix Acc: {valid_pix_acc:.4f}, Valid mIoU: {valid_miou:.4f}")
    
    if valid_miou > best_miou or (valid_miou == best_miou and valid_pix_acc > best_pixel_acc):
        best_miou = valid_miou
        best_pixel_acc = valid_pix_acc
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'best_miou': best_miou,
            'best_pixel_acc': best_pixel_acc,
        }, f"{BASE_DIR}/models/best_cgan_model_{epoch+1}.pth")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break


generator.eval()
discriminator.eval()
test_loss, test_pix_acc, test_miou = 0.0, 0.0, 0.0
generated_ki67 = []
real_ki67 = []
he_images_list = []

virtualization_dir = os.path.join(BASE_DIR, "virtualized_results")
os.makedirs(virtualization_dir, exist_ok=True)

with torch.no_grad():
    for idx, (he_imgs, ki67_imgs) in enumerate(test_loader):
        he_imgs = he_imgs.to(device)
        ki67_imgs = ki67_imgs.to(device)
        

        noise = torch.randn(he_imgs.size(0), 3, target_shape[0], target_shape[1], device=device)
        fake_ki67 = generator(noise, he_imgs)
        fake_ki67_clamped = torch.clamp(fake_ki67, -1, 1)
        fake_ki67_01 = torch.clamp((fake_ki67_clamped + 1) / 2, 0, 1)
        
        ki67_01 = torch.clamp((ki67_imgs + 1) / 2, 0, 1) if ki67_imgs.min() < 0 else ki67_imgs
        

        he_images_list.append(he_imgs.cpu())
        generated_ki67.append(fake_ki67_01.cpu())
        real_ki67.append(ki67_01.cpu())
        

        test_loss += criterion_l1(fake_ki67_01, ki67_01).item() * he_imgs.size(0)
        test_pix_acc += pixel_accuracy(fake_ki67_01, ki67_01) * he_imgs.size(0)
        test_miou += m_iou(fake_ki67_01, ki67_01) * he_imgs.size(0)
        

        for i in range(he_imgs.size(0)):
            dataset_idx = idx * test_loader.batch_size + i
            if dataset_idx < len(test_dataset.img_paths):
                orig_path = test_dataset.img_paths[dataset_idx]
                file_name = f"virtualized_{Path(orig_path).name}"
                save_image(
                    fake_ki67_01[i].cpu(),
                    os.path.join(virtualization_dir, file_name),
                    normalize=True
                )
        

        if idx % 10 == 0:
            for i in range(min(2, he_imgs.size(0))):

                he_display = torch.clamp((he_imgs[i].cpu() + 1) / 2, 0, 1) if he_imgs.min() < 0 else he_imgs[i].cpu()
                triplet = torch.cat([
                    he_display,
                    fake_ki67_01[i].cpu(),
                    ki67_01[i].cpu()
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
print(f"L1 Loss: {test_loss:.4f}")
print(f"Modelo treinado com {num_classes} classes, batch de tamanho {batch_size}\n")
print("Testing complete.")
