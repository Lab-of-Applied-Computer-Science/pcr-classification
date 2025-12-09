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

stain = "KI67"
# stain = "PHH3"

metrica = "SSIM"

BASE_DIR = f"/scratch/henrique.colonese/Virtual_Staining/{stain}/{metrica}/results_{date_str}"
os.makedirs(f"{BASE_DIR}", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/results", exist_ok=True)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.resblock1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.resblock2 = ResBlock(out_channels, out_channels, time_emb_dim)

        self.downsample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if downsample else nn.Identity()

    def forward(self, x, time_emb):
        h = self.resblock1(x, time_emb)
        h = self.resblock2(h, time_emb)

        if self.downsample:
            return self.downsample_conv(h), h
        return h, h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.resblock1 = ResBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.resblock2 = ResBlock(out_channels, out_channels, time_emb_dim)
        self.upsample_conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1) if upsample else nn.Identity()

    def forward(self, x, skip, time_emb):
        if self.upsample:
            x = self.upsample_conv(x)
        h = torch.cat([x, skip], dim=1)
        h = self.resblock1(h, time_emb)
        h = self.resblock2(h, time_emb)

        return h

class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=3, cond_channels=3, time_emb_dim=256, base_channels=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        

        self.init_conv = nn.Conv2d(img_channels + cond_channels, base_channels, 3, padding=1)
        

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        

        self.middle = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim),
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim),
        )
        

        self.up1 = UpBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up3 = UpBlock(base_channels * 2, base_channels, time_emb_dim)
        

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1),
        )

    def forward(self, x, condition, timestep):

        time_emb = self.time_mlp(timestep)
        

        h = torch.cat([x, condition], dim=1)
        h = self.init_conv(h)
        

        h, skip1 = self.down1(h, time_emb)
        h, skip2 = self.down2(h, time_emb)
        h, skip3 = self.down3(h, time_emb)
        

        for layer in self.middle:
            h = layer(h, time_emb)
        

        h = self.up1(h, skip3, time_emb)
        h = self.up2(h, skip2, time_emb)
        h = self.up3(h, skip1, time_emb)
        
        return self.output_conv(h)

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t.to(device) * x_0 + sqrt_one_minus_alphas_cumprod_t.to(device) * noise, noise

def sample_ddpm(model, condition, noise_scheduler, device, num_inference_steps=50):
    """DDPM sampling function"""
    model.eval()
    batch_size = condition.shape[0]
    

    img = torch.randn((batch_size, 3, condition.shape[2], condition.shape[3]), device=device)
    

    timesteps = torch.linspace(noise_scheduler.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).repeat(batch_size)
        
        with torch.no_grad():

            noise_pred = model(img, condition, t_batch)
            

            alpha_t = noise_scheduler.get_index_from_list(noise_scheduler.alphas, t_batch, img.shape)
            alpha_t_cumprod = noise_scheduler.get_index_from_list(noise_scheduler.alphas_cumprod, t_batch, img.shape)
            beta_t = noise_scheduler.get_index_from_list(noise_scheduler.betas, t_batch, img.shape)
            sqrt_one_minus_alpha_cumprod_t = noise_scheduler.get_index_from_list(
                noise_scheduler.sqrt_one_minus_alphas_cumprod, t_batch, img.shape
            )
            

            pred_original_sample = (img - sqrt_one_minus_alpha_cumprod_t * noise_pred) / torch.sqrt(alpha_t_cumprod)
            

            pred_original_sample_coeff = torch.sqrt(noise_scheduler.get_index_from_list(
                noise_scheduler.alphas_cumprod_prev, t_batch, img.shape
            )) * beta_t / (1 - alpha_t_cumprod)
            
            current_sample_coeff = torch.sqrt(alpha_t) * (1 - noise_scheduler.get_index_from_list(
                noise_scheduler.alphas_cumprod_prev, t_batch, img.shape
            )) / (1 - alpha_t_cumprod)
            

            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * img
            

            if i < len(timesteps) - 1:
                noise = torch.randn_like(img)
                variance = noise_scheduler.get_index_from_list(noise_scheduler.posterior_variance, t_batch, img.shape)
                pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
            
            img = pred_prev_sample
    
    return img

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
    df['corrected_KI67'] = df['KI-67_Path'].apply(get_path)
    

    if metric == "MEDIA":

        valid_pairs = df[
            ((df['Similarity'] / 255000000) > min_miou) &
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]
    
    elif metric == "SSIM":

        valid_pairs = df[
            (df['SSIM']  > min_miou) &
            (df['corrected_HES'].notnull()) &
            (df['corrected_KI67'].notnull())
        ]
    
    elif metric == "MI":

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

dataset_dir = '/home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/bcs-kedro/data/02_intermediate/spatial-attention-pcr/'
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
    csv_path='/home_cerberus/speed/henrique.colonese/Virtual_Staining/KI-67_Similarity_Full_Dataset.csv',
    metric="none",
    min_miou=0.6
)

full_df = pd.read_csv('/home_cerberus/speed/henrique.colonese/Virtual_Staining/KI-67_Similarity_Full_Dataset.csv')
full_df['corrected_HES'] = full_df['HES_Path'].apply(get_path)
full_df['corrected_KI67'] = full_df['KI-67_Path'].apply(get_path)


print("Splitting dataset paths...")

img_train, img_valid, img_test, target_train, target_valid, target_test = split_by_patient(
    img_paths, 
    target_paths, 
    test_size=0.1, 
    val_size=0.15, 
    random_state=42
)

save_splits((img_train, img_valid, img_test, target_train, target_valid, target_test))


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


model = ConditionalUNet(img_channels=3, cond_channels=3, time_emb_dim=256, base_channels=64).to(device)
noise_scheduler = NoiseScheduler(timesteps=500)

print("Diffusion model initialized")

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_ssim = SSIM(data_range=1.0, kernel_size=11, reduction='elementwise_mean').to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print("Entering diffusion training loop")
scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_pix_acc, train_miou = 0.0, 0.0
    
    for he_images, ki67_images in train_loader:
        he_images = he_images.to(device)
        ki67_images = ki67_images.to(device)
        batch_size = he_images.size(0)
        
        optimizer.zero_grad()
        
        with autocast():
            t = torch.randint(0, noise_scheduler.timesteps, (batch_size,), device=device).long()
            
            x_noisy, noise = noise_scheduler.forward_diffusion_sample(ki67_images, t, device)
            
            noise_pred = model(x_noisy, he_images, t)
            loss_noise = criterion_mse(noise_pred, noise)
            

            if epoch > 10:
                with torch.no_grad():

                    alpha_t = noise_scheduler.get_index_from_list(noise_scheduler.alphas, t, x_noisy.shape)
                    sqrt_one_minus_alpha_t = noise_scheduler.get_index_from_list(
                        noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
                    )
                    x_pred = (x_noisy - sqrt_one_minus_alpha_t * noise_pred) / torch.sqrt(alpha_t)
                    x_pred = torch.clamp(x_pred, 0, 1)
                
                loss_l1 = criterion_l1(x_pred, ki67_images)
                loss_ssim = 1 - criterion_ssim(x_pred, ki67_images)
                
                total_loss = loss_noise + 0.5 * loss_l1 + 0.1 * loss_ssim
            else:
                total_loss = loss_noise
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if epoch > 10:
            with torch.no_grad():
                batch_pix_acc = pixel_accuracy(x_pred, ki67_images)
                batch_miou = m_iou(x_pred, ki67_images)
                train_pix_acc += batch_pix_acc * batch_size
                train_miou += batch_miou * batch_size
        
        train_loss += total_loss.item() * batch_size
    

    train_loss /= len(train_loader.dataset)
    if epoch > 10:
        train_pix_acc /= len(train_loader.dataset)
        train_miou /= len(train_loader.dataset)
    

    model.eval()
    valid_loss, valid_pix_acc, valid_miou = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for he_images, ki67_images in valid_loader:
            he_images = he_images.to(device)
            ki67_images = ki67_images.to(device)
            batch_size = he_images.size(0)
            

            generated_ki67 = sample_ddpm(model, he_images, noise_scheduler, device, num_inference_steps=50)
            generated_ki67 = torch.clamp(generated_ki67, 0, 1)
            

            valid_loss += criterion_l1(generated_ki67, ki67_images).item() * batch_size
            valid_pix_acc += pixel_accuracy(generated_ki67, ki67_images) * batch_size
            valid_miou += m_iou(generated_ki67, ki67_images) * batch_size
    
    valid_loss /= len(valid_loader.dataset)
    valid_pix_acc /= len(valid_loader.dataset)
    valid_miou /= len(valid_loader.dataset)
    
    scheduler.step(valid_loss)
    
    if epoch <= 10:
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f}, Pix Acc: {valid_pix_acc:.4f}, mIoU: {valid_miou:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, "
              f"Pix Acc: {train_pix_acc:.4f}, mIoU: {train_miou:.4f} | "
              f"Valid Loss: {valid_loss:.4f}, Pix Acc: {valid_pix_acc:.4f}, mIoU: {valid_miou:.4f}")
    

    if valid_miou > best_miou or (valid_miou == best_miou and valid_pix_acc > best_pixel_acc):
        best_miou = valid_miou
        best_pixel_acc = valid_pix_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'best_pixel_acc': best_pixel_acc,
            'noise_scheduler': noise_scheduler
        }, f"{BASE_DIR}/models/best_diffusion_model_{epoch+1}.pth")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break




model.eval()
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
        

        fake_ki67 = sample_ddpm(model, he_imgs, noise_scheduler, device, num_inference_steps=100)
        fake_ki67 = torch.clamp(fake_ki67, 0, 1)
        

        he_images.append(he_imgs.cpu())
        generated_ki67.append(fake_ki67.cpu())
        real_ki67.append(ki67_imgs.cpu())
        
        test_loss += criterion_l1(fake_ki67, ki67_imgs).item() * he_imgs.size(0)
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
print(f"L1 Loss: {test_loss:.4f}")
print(f"Modelo treinado com {num_classes} classes, batch de tamanho {batch_size}\n")
print("Testing complete.")


