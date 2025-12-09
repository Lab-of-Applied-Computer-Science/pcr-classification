import logging
import os
from datetime import datetime
import gc
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
import psutil
import resource
import signal
from clearml import Task
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from .custom_callbacks import MetricsCallback
import matplotlib.pyplot as plt
from keras.optimizers import SGD  # Import SGD optimizer
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import concatenate, UpSampling2D, Reshape
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import AdamW
from tensorflow.python.framework import ops
logger = logging.getLogger(__name__)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

mixed_precision.set_global_policy('mixed_float16')


# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU: {gpu}")
            print(f"Details: {tf.config.experimental.get_device_details(gpu)}")
            
            tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=14 * 1024)]  # 16GB
            )
    except RuntimeError as e:
        print("GPU configuration error:", e)

def filter_missing_files(df):

    valid_rows = []
    
    for _, row in df.iterrows():
        # Process paths exactly like in your training pipeline
        try:
            paths = {
                'hes': row["path_phh3"].replace("PHH3", "HES"),  # String replacement
                'ki67': row["path_ki67"],
                'phh3': row["path_phh3"],
                'att_map': row['att_map_path']
            }
            
            # Check all files exist
            if all(os.path.exists(p) for p in paths.values()):
                valid_rows.append(row)
                
        except (KeyError, AttributeError) as e:
            print(f"Skipping row due to error: {e}")
            continue
            
    # Return new DataFrame with only valid rows
    filtered_df = pd.DataFrame(valid_rows)
    
    # Log statistics
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} rows due to missing files")
    
    return filtered_df

    
def create_safe_dataset(df, batch_size=2, target_size=(512, 512)):
    """Memory-constrained pipeline with emergency safeguards"""
    
    # Emergency memory checker
    class MemoryGuard(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if psutil.virtual_memory().percent > 95:
                self.model.stop_training = True
                raise RuntimeError("Emergency stop: memory limit exceeded")

    # Optimized image loader with hard limits
    @tf.function
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        return tf.cast(img, tf.float16) / 255.0  # float16 saves memory

    # Safe NPZ loader
    def load_att_map(npz_path):
        def _load_att_map(npz_path):
            try:
                with np.load(npz_path) as data:
                    att_map = data['attmap']
                    if att_map.ndim == 2:
                        att_map = np.repeat(att_map[..., np.newaxis], 3, axis=-1)
                    return tf.image.resize(att_map, target_size) / (np.max(att_map) + 1e-6)
            except Exception as e:
                print(f"Error loading {npz_path}: {str(e)}")
                return np.zeros((*target_size, 3), dtype=np.float16)
        
        att_map = tf.numpy_function(
            _load_att_map,
            [npz_path],
            tf.float16  # float16 saves memory
        )
        att_map.set_shape((*target_size, 3))
        return att_map

    # Process single sample at a time
    def process_row(row):
        try:
            # Process with memory checks
            if psutil.virtual_memory().percent > 90:
                raise MemoryError("Preemptive memory stop")
                
            # Load one image at a time
            img = load_image(row['path_phh3'])
            att_map = load_att_map(row['att_map_path'])
            
            # Combined operations
            combined = tf.concat([img * att_map], axis=-1)  # Simplified for demo
            combined.set_shape((*target_size, 3))  # Reduced channels for safety
            
            label = tf.cast(tf.strings.regex_full_match(row['target'], "pcr"), tf.int32)
            return combined, label
            
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            return tf.zeros((*target_size, 3)), tf.constant(0, dtype=tf.int32)

    # Create dataset with strict memory controls
    options = tf.data.Options()
    options.experimental_optimization.autotune = False  # Disable auto-tuning
    options.experimental_optimization.map_and_batch_fusion = False
    
    dataset = tf.data.Dataset.from_generator(
        lambda: (dict(row) for _, row in df.iterrows()),
        output_types={k: tf.string for k in df.columns},
        output_shapes={k: [] for k in df.columns}
    )
    
    dataset = dataset.with_options(options)
    dataset = dataset.map(process_row, num_parallel_calls=1)  # Single-threaded
    dataset = dataset.batch(batch_size)
    
    return dataset

# @tf.function
# def preprocess_path(path, is_hes=False):
#     """Centralized path preprocessing"""
#     if is_hes:
#         path = tf.strings.regex_replace(path, "PHH3", "HES")
#     path = tf.strings.regex_replace(path, "03_primary", "02_intermediate")
#     path = tf.strings.regex_replace(path, "-deconv", "")
#     return path

@tf.function
def preprocess_path(path, is_hes=False):
    if is_hes:
        path = tf.strings.regex_replace(path, "PHH3", "HES")
    return tf.strings.regex_replace(tf.strings.regex_replace(path, "03_primary", "02_intermediate"), "-deconv", "")

@tf.function
def load_image(path, target_size=(256, 256)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    return tf.cast(img, tf.float16) / 255.0

@tf.function
def load_att_map(npz_path):
    npz_path = tf.strings.regex_replace(npz_path, "/snfs1/speed/", "/scratch/")            
    npz_path = tf.strings.regex_replace(npz_path, "bcs-kedro/", "")  
    npz_path = tf.strings.regex_replace(npz_path, "ihc-only-tensor", "tiles")
    def _load_npz(npz_path_bytes):
        npz_path_str = npz_path_bytes.decode('utf-8')
        data = np.load(npz_path_str)
        return data["attmap"].astype(np.float32)


    att_map = tf.numpy_function(_load_npz, [npz_path], tf.float32)
    att_map.set_shape([1000, 1000])
    att_map = tf.expand_dims(att_map, axis=-1)
    att_map = tf.repeat(att_map, repeats=3, axis=-1)
    att_map = tf.image.resize(att_map, (256, 256))
    normalized = att_map / (tf.reduce_max(att_map) + 1e-6)
    return tf.cast(normalized, tf.float16)

def process_row(row):
    target_size=(256, 256)
    try:
        # path_ki67 = preprocess_path(row["path_ki67"])
        # path_phh3 = preprocess_path(row["path_phh3"])
        path_hes = preprocess_path(row["path_phh3"], is_hes=True)
        att_map_path = row['att_map_path']
        
        hes = load_image(path_hes)
        # ki67 = load_image(path_ki67)
        # phh3 = load_image(path_phh3)
        att_map = load_att_map(att_map_path)
            
        # tf.debugging.assert_all_finite(att_map, "Attention map has NaN/Inf")
            
        hes.set_shape((*target_size, 3))
        # ki67.set_shape((*target_size, 3))
        # phh3.set_shape((*target_size, 3))
        att_map.set_shape((*target_size, 3))
                        
        alpha = 0.01
        hes_att = alpha * hes + (1-alpha) * (hes * att_map)
        # ki67_att = alpha * ki67 + (1-alpha) * (ki67 * att_map)
        # phh3_att = alpha * phh3 + (1-alpha) * (phh3 * att_map)
            
        # Concatenate results
        # combined = tf.concat([hes], axis=-1)
        # combined.set_shape((*target_size, 3))
            
        # Get label
        label = tf.cast(tf.strings.regex_full_match(row['target'], "pcr"), tf.int32)
        return hes_att, label
            
    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return tf.zeros((*target_size, 3), dtype=tf.float16), tf.constant(0, tf.int32)

def create_optimized_dataset(df, batch_size=2, target_size=(256, 256), repeat=False):
    # tf.keras.backend.clear_session()
    # gc.collect()
    logger.debug("entrou create")
    options = tf.data.Options()
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.with_options(options)
    ds = ds.map(process_row, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds

def debug_visualization(df, sample_indices, output_dir="debug_visualizations"):
    """
    Generate visualizations of original images, attention maps, and their product
    
    Args:
        df: DataFrame with image paths
        sample_indices: List of indices to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    target_size = (256, 256)
    
    def preprocess_path(path, is_hes=False):
        if is_hes:
            path = tf.strings.regex_replace(path, "PHH3", "HES").numpy().decode()
        else:
            path = path.numpy().decode()
        path = path.replace("03_primary", "02_intermediate")
        path = path.replace("-deconv", "")
        return path
    
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        return tf.cast(img, tf.float32) / 255.0
    
    def load_att_map(npz_path):
        try:
            # First try loading as NPZ file
            with np.load(npz_path.numpy().decode()) as data:
                keys = list(data.keys())
                if keys:
                    att_map = data[keys[0]]
                else:
                    raise ValueError("Empty NPZ file")
        except:
            try:
                # If that fails, try loading as serialized tensor
                raw_data = tf.io.read_file(npz_path)
                att_map = tf.io.parse_tensor(raw_data, out_type=tf.float32).numpy()
            except:
                print(f"Failed to load attention map: {npz_path}")
                att_map = np.zeros((*target_size, 1), dtype=np.float32)
        
        # Handle different shapes
        if att_map.ndim == 2:
            att_map = np.expand_dims(att_map, axis=2)
            att_map = np.repeat(att_map, 3, axis=2)
        elif att_map.ndim == 3 and att_map.shape[2] == 1:
            att_map = np.repeat(att_map, 3, axis=2)
        
        # Resize and normalize
        att_map = tf.image.resize(att_map, target_size).numpy()
        if np.max(att_map) > 0:
            att_map = att_map / np.max(att_map)
        
        return att_map
    
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Get paths
        path_ki67 = preprocess_path(tf.convert_to_tensor(row["path_ki67"]))
        path_phh3 = preprocess_path(tf.convert_to_tensor(row["path_phh3"]))
        path_hes = preprocess_path(tf.convert_to_tensor(row["path_phh3"]), is_hes=True)
        att_map_path = tf.convert_to_tensor(row['att_map_path'])
        
        # Load images
        hes = load_image(path_hes).numpy()
        ki67 = load_image(path_ki67).numpy()
        phh3 = load_image(path_phh3).numpy()
        att_map = load_att_map(att_map_path)
        
        # weighted - 1
        alpha = 0.7

        hes_att = alpha * hes + (1-alpha) * (hes * att_map)
        ki67_att = alpha * ki67 + (1-alpha) * (ki67 * att_map)
        phh3_att = alpha * phh3 + (1-alpha) * (phh3 * att_map)
        
        # enhanced - 4
        # enhanced_he = np.clip(hes * 1.5, 0.0, 1.0)
        # hes_att = hes * (1 - att_map) + enhanced_he * att_map
        
        # enhanced_ki67 = np.clip(ki67 * 1.5, 0.0, 1.0)
        # ki67_att = ki67 * (1 - att_map) + enhanced_ki67 * att_map
        
        # enhanced_phh3 = np.clip(phh3 * 1.5, 0.0, 1.0)
        # phh3_att = phh3 * (1 - att_map) + enhanced_phh3 * att_map
        
        # Create visualization
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        
        # Original images
        axs[0, 0].imshow(hes)
        axs[0, 0].set_title("HES Original")
        axs[1, 0].imshow(ki67)
        axs[1, 0].set_title("KI67 Original")
        axs[2, 0].imshow(phh3)
        axs[2, 0].set_title("PHH3 Original")
        
        # Attention map
        axs[0, 1].imshow(att_map)
        axs[0, 1].set_title("Attention Map")
        # Add a heatmap version of attention
        axs[1, 1].imshow(att_map[:,:,0], cmap='hot')
        axs[1, 1].set_title("Attention Map (Heatmap)")
        # Add a histogram of attention values
        axs[2, 1].hist(att_map.flatten(), bins=50)
        axs[2, 1].set_title("Attention Values Distribution")
        
        # Images after attention
        axs[0, 2].imshow(hes_att)
        axs[0, 2].set_title("HES × Attention")
        axs[1, 2].imshow(ki67_att)
        axs[1, 2].set_title("KI67 × Attention")
        axs[2, 2].imshow(phh3_att)
        axs[2, 2].set_title("PHH3 × Attention")
        
        # Difference images
        axs[0, 3].imshow(np.abs(hes - hes_att) * 5)  # Amplify differences for visibility
        axs[0, 3].set_title("HES Difference (×5)")
        axs[1, 3].imshow(np.abs(ki67 - ki67_att) * 5)
        axs[1, 3].set_title("KI67 Difference (×5)")
        axs[2, 3].imshow(np.abs(phh3 - phh3_att) * 5)
        axs[2, 3].set_title("PHH3 Difference (×5)")
        
        # Add stats
        plt.suptitle(f"Sample {idx} | Attention Map Stats: Min={att_map.min():.4f}, Max={att_map.max():.4f}, Mean={att_map.mean():.4f}")
        
        for ax in axs.flatten():
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the combined 9D image as separate channels
        combined = np.concatenate([hes_att, ki67_att, phh3_att], axis=2)
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for i in range(9):
            row, col = i // 3, i % 3
            axs[row, col].imshow(combined[:, :, i], cmap='viridis')
            img_type = ["HES", "KI67", "PHH3"][i // 3]
            channel = ["R", "G", "B"][i % 3]
            axs[row, col].set_title(f"{img_type} × Attention ({channel} channel)")
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_{idx}_9d_channels.png", dpi=150, bbox_inches='tight')
        plt.close()    

def load_split_datasets():
    """
    Load pre-split training and validation datasets from files.
    """
    logger.debug("Loading pre-split datasets from 'data_splits/' directory.")
    train_df = pd.read_csv("data_splits/train_data.csv")
    val_df = pd.read_csv("data_splits/val_data.csv")
    return train_df, val_df


def setup_clearml(project_name, task_name, tags):
    logger.debug("Setting up ClearML task")
    task = Task.init(
        project_name=project_name, task_name=task_name, tags=tags, output_uri=True
    )
    return task

def build_unet_model():
    num_classes = 1
    # Input layer
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    # --- Encoder (Downsampling Path) ---
    
    # Initial 9-channel adaptation
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Contracting Path
    def conv_block(x, filters, kernel_size=(3,3), padding='same', strides=1):
        x = tf.keras.layers.SeparableConv2D(  # Changed from Conv2D
            filters, kernel_size, 
            padding=padding, 
            strides=strides,
            depthwise_regularizer=l1_l2(l1=1e-5, l2=3e-4)
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    # Store skip connections
    skip_connections = []
    
    # Downsample blocks
    def downsample_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        skip = x
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x, skip
    
    # Encoder blocks
    x, skip1 = downsample_block(x, 32)  # 256x256 → 128x128
    x, skip2 = downsample_block(x, 64)  # 128x128 → 64x64
    x, skip3 = downsample_block(x, 128) # 64x64 → 32x32
    
    # --- Bottleneck ---
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # --- Decoder (Upsampling Path) ---
    def upsample_block(x, skip, filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(filters, (2, 2), padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    # Reverse skip connections for decoder
    # skip_connections.reverse()
    
    # Decoder blocks
    x = upsample_block(x, skip3, 128)  # 32x32 → 64x64
    x = upsample_block(x, skip2, 64)   # 64x64 → 128x128
    x = upsample_block(x, skip1, 32)   # 128x128 → 256x256
    
    # --- Classification Head ---
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(512, activation='relu', 
    #           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    # x = Dropout(0.3)(x)
    # x = Dense(256, activation='relu', 
    #           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    # x = Dropout(0.2)(x)
    
    # outputs = Dense(num_classes, activation='sigmoid')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_unet_model_3d():
    num_classes = 1
    # Input layer
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    # --- Encoder (Downsampling Path) ---
    
    # Initial 9-channel adaptation
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Contracting Path
    def conv_block(x, filters, kernel_size=(3,3), padding='same', strides=1):
        x = tf.keras.layers.SeparableConv2D(  # Changed from Conv2D
            filters, kernel_size, 
            padding=padding, 
            strides=strides,
            depthwise_regularizer=l1_l2(l1=1e-5, l2=3e-4)
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    # Store skip connections
    skip_connections = []
    
    # Downsample blocks
    def downsample_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        skip = x
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x, skip
    
    # Encoder blocks
    x, skip1 = downsample_block(x, 32)  # 256x256 → 128x128
    x, skip2 = downsample_block(x, 64)  # 128x128 → 64x64
    x, skip3 = downsample_block(x, 128) # 64x64 → 32x32
    
    # --- Bottleneck ---
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # --- Decoder (Upsampling Path) ---
    def upsample_block(x, skip, filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(filters, (2, 2), padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    
    # Decoder blocks
    x = upsample_block(x, skip3, 128)  # 32x32 → 64x64
    x = upsample_block(x, skip2, 64)   # 64x64 → 128x128
    x = upsample_block(x, skip1, 32)   # 128x128 → 256x256
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
def configure_training():
    # Enable mixed precision with memory optimizations
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    # Configure optimizer with gradient clipping
    optimizer = AdamW(
        learning_rate=3e-5,
        weight_decay=1e-4,
        global_clipnorm=1.0,
    )
    
    return mixed_precision.LossScaleOptimizer(
        optimizer,
        dynamic=True  # This is better for memory management
    ), tf.keras.losses.BinaryCrossentropy()

def build_resnet34_classifier():
    num_classes = 1
    inputs = tf.keras.Input(shape=(256, 256, 3))

    def residual_block(x, filters, reduction=False):
        """Lightweight residual block with depthwise separable convs"""
        shortcut = x
        stride = 2 if reduction else 1
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # Depthwise separable convolution path
        x = tf.keras.layers.SeparableConv2D(
            filters, (3,3), padding='same', 
            strides=stride, depth_multiplier=1
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.SeparableConv2D(
            filters, (3,3), padding='same',
            depth_multiplier=1
        )(x)
        
        # Shortcut connection for dimension matching
        if reduction:
            shortcut = tf.keras.layers.SeparableConv2D(
                filters, (1,1), strides=2
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            
        x = tf.keras.layers.Add()([x, shortcut])
        return x

    # Initial adaptation for 3-channel input (same as UNet)
    x = tf.keras.layers.Conv2D(
        64, (7,7), padding='same', 
        kernel_regularizer=tf.keras.regularizers.l2(3e-4)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual stages for 32-layer depth
    filters = [64, 128, 256]
    repeats = [3, 4, 2]  # Total layers: 3*(1+3) + 4*2 + 2*2 = 3*4 + 4*2 + 2*2 = 12 + 8 +4 =24? +8 initial layers=32
    
    for stage in range(3):
        # First block in stage might need downsampling
        x = residual_block(x, filters[stage], reduction=(stage > 0))
        # Additional blocks
        for _ in range(repeats[stage]):
            x = residual_block(x, filters[stage])

    # Final classification head (matches UNet output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        64, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(3e-4)
    )(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='sigmoid',
        dtype='float32'  # Critical for mixed precision
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# def build_resnet34_classifier():
#     """Canonical ResNet-34 (4 stages, 3-3-6-3 blocks) for 256×256×9 input."""
    
#     weight_decay = 3e-4
#     num_classes  = 1
#     inputs = tf.keras.Input(shape=(256, 256, 9))
    
#     # ─────────────────────────────
#     # Helper: pre-activation basic block
#     # ─────────────────────────────
#     def residual_block(x, filters, stride=1):
#         shortcut = x
        
#         # Pre-activation
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)
        
#         # First 3×3 conv
#         x = tf.keras.layers.Conv2D(
#             filters, 3, strides=stride, padding='same',
#             use_bias=False,
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
#         )(x)
        
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)
        
#         # Second 3×3 conv
#         x = tf.keras.layers.Conv2D(
#             filters, 3, padding='same', use_bias=False,
#             kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
#         )(x)
        
#         # Projection shortcut if spatial or channel dims change
#         if stride != 1 or shortcut.shape[-1] != filters:
#             shortcut = tf.keras.layers.Conv2D(
#                 filters, 1, strides=stride, use_bias=False,
#                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
#             )(shortcut)
#             shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
#         x = tf.keras.layers.Add()([x, shortcut])
#         return x
    
#     # ─────────────────────────────
#     # Stem
#     # ─────────────────────────────
#     x = tf.keras.layers.Conv2D(
#         64, 7, strides=2, padding='same', use_bias=False,
#         kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
#     )(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
#     # ─────────────────────────────
#     # Residual stages (64,128,256,512) with 3-4-6-3 blocks
#     # ─────────────────────────────
#     filters  = [64, 128, 256, 512]
#     n_blocks = [3,  4,   6,   3]
    
#     for f, n in zip(filters, n_blocks):
#         # First block in each stage – stride 2 except for the first stage
#         stride = 1 if f == 64 else 2
#         x = residual_block(x, f, stride=stride)
#         for _ in range(n - 1):
#             x = residual_block(x, f, stride=1)
    
#     # Final BN + ReLU (ResNet-v2 style)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
    
#     # ─────────────────────────────
#     # Classification head
#     # ─────────────────────────────
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     outputs = tf.keras.layers.Dense(
#         num_classes,
#         activation='sigmoid',          # binary output
#         kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
#     )(x)
    
#     return tf.keras.Model(inputs, outputs)


class MemoryMonitor(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.85):
        super().__init__()
        self.threshold = threshold
    
    def on_train_begin(self, logs=None):
        self.batch_memory_usage = []
    
    def on_batch_end(self, batch, logs=None):
        # Get detailed memory stats
        stats = tf.config.experimental.get_memory_info('GPU:0')
        used = stats['current'] / (1024**3)
        total = stats['peak'] / (1024**3)
        self.batch_memory_usage.append(used)
        
        logs = logs or {}
        logs['gpu_memory_used_gb'] = used
        logs['gpu_memory_ratio'] = used/total
        
        if used > self.threshold * total:
            print(f"\nMemory threshold ({self.threshold*100:.0f}%) exceeded - clearing session")
            tf.keras.backend.clear_session()
            gc.collect()
    
    def on_epoch_end(self, epoch, logs=None):
        # Log memory usage statistics
        if self.batch_memory_usage:
            avg_mem = np.mean(self.batch_memory_usage)
            max_mem = np.max(self.batch_memory_usage)
            print(f"\nEpoch {epoch+1} Memory Stats - Avg: {avg_mem:.2f}GB, Max: {max_mem:.2f}GB")
            self.batch_memory_usage = []

###########################
#         Nodes           #
# ###########################

def split_dataset(attention_map_tile_metadata: pd.DataFrame, mapping_split_train_val: str, test_size: float = 0.2, random_state: int = 42, balance_classes: bool = True):
    """
    Simple stratified group split: All tiles from same img_id stay together
    """
    group_col= 'img_id'
    target_col= 'target'
    
    if attention_map_tile_metadata.empty:
        raise ValueError("Input DataFrame is empty!")
    
    np.random.seed(random_state)
    
    # Get unique image groups with their target
    groups = attention_map_tile_metadata[[group_col, target_col]].drop_duplicates()
    pos_groups = groups[groups[target_col] == "pcr"]
    neg_groups = groups[groups[target_col] == "non-pcr"]
    
    print(f"POS GROUPS: {len(pos_groups)}")
    print(f"NEG GROUPS: {len(neg_groups)}")
    
    # Shuffle groups within each class
    pos_groups = shuffle(pos_groups, random_state=random_state)
    neg_groups = shuffle(neg_groups, random_state=random_state)
    
    # Initialize splits
    train_groups = []
    val_groups = []
    
    # Define target balance thresholds
    target_ratio = len(pos_groups) / len(groups)  # Original positive class ratio
    current_ratio = lambda t,v: len(t) / (len(t)+len(v))  # Simplified ratio tracking
    
    # Distribute groups while preserving ratio
    for class_groups in [pos_groups, neg_groups]:
        split_idx = int(len(class_groups) * (1 - test_size))
        train_groups.extend(class_groups[group_col].values[:split_idx])
        val_groups.extend(class_groups[group_col].values[split_idx:])
    
    # Final splits
    train_df = attention_map_tile_metadata[attention_map_tile_metadata[group_col].isin(train_groups)].sample(frac=1)  # Shuffle tiles
    val_df = attention_map_tile_metadata[attention_map_tile_metadata[group_col].isin(val_groups)].sample(frac=1)
    
    # Verify no leakage
    common = set(train_df[group_col]).intersection(val_df[group_col])
    print(f"Groups in both splits: {len(common)} (should be 0)")
    
        # Save the splits to files
    os.makedirs(mapping_split_train_val, exist_ok=True)
    train_df.to_csv(f"{mapping_split_train_val}/train_data.csv", index=False)
    val_df.to_csv(f"{mapping_split_train_val}/val_data.csv", index=False)

    return train_df, val_df
    


def train_cnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_classes: int,
    model_checkpoint_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    metrics_output_dir: str,
    clearml_params: dict,
    use_pretrained_weights: bool = True,
):
    # Initialize logging and ClearML
    logger.debug("ClearML Params: %s", clearml_params)
    # setup_clearml(
    #     project_name=clearml_params["project_name"],
    #     task_name=clearml_params["task_name"],
    #     tags=clearml_params["tags"],
    # )

    tf.keras.backend.clear_session()
    gc.collect()
    
    # Create model
    # model = build_unet_model()
    
    model = build_resnet34_classifier()
    optimizer, loss_fn = configure_training()
    
    # Compile model with enhanced metrics
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            # tf.keras.metrics.AUC(name='pr_auc', curve='PR')
        ]
    )

    # debug_visualization(train_df, sample_indices=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], output_dir="debug_visualizations")
    
    logger.debug("pre dataset gen")
    train_gen = create_optimized_dataset(train_df, batch_size=batch_size, repeat=True)
    val_gen = create_optimized_dataset(val_df, batch_size=batch_size, repeat=False)
    logger.debug("pos dataset gen")
    
    # Calculate steps
    train_steps = int(np.ceil(len(train_df) / batch_size))
    val_steps = int(np.ceil(len(val_df) / batch_size))

    os.makedirs(model_checkpoint_path, exist_ok=True)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            # os.path.join(model_checkpoint_path, "unet_pcr_clf_best.h5"),
            os.path.join(model_checkpoint_path, "unet_pcr_clf_best.h5"),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
            verbose=1,
            # save_weights_only=True
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.5,
        #     patience=5,
        #     min_lr=1e-6,
        #     verbose=1
        # ),
        TensorBoard(
            log_dir=metrics_output_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch"
        ),
        MemoryMonitor(),
    ]

    # Calculate class weights for imbalanced data
    class_counts = train_df['target'].value_counts()
    class_weight = {0: 1.0, 1: 1.0}
    logger.debug(f"Class Weights: {class_weight}")
    
    # Train the model
    logger.debug("Training model")
    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Explicit cleanup if training fails
        tf.keras.backend.clear_session()
        gc.collect()
        raise

    os.makedirs(metrics_output_dir, exist_ok=True)
    history_file = os.path.join(metrics_output_dir, "training_history.csv")
    pd.DataFrame(history.history).to_csv(history_file, index=False)

    tf.keras.backend.clear_session()
    gc.collect()
    
    return val_gen, val_steps
    
def test_cnn(
    model_checkpoint_path: str,
    val_gen: tf.data.Dataset,  # Changed from val_gen to test_gen for clarity
    val_steps: int,
    metrics_output_dir: str,
    threshold_tuning: bool = True  # Add option for optimal threshold finding
):
    """
    Enhanced testing function with:
    - Better performance discrepancy analysis
    - Threshold optimization
    - Comprehensive metrics reporting
    - Detailed visualizations
    """
    logger.debug("Testing model with enhanced evaluation")

    # Load the best model
    best_model_path = os.path.join(model_checkpoint_path, "unet_pcr_clf_best.h5")
    logger.debug(f"Loading best model from: {best_model_path}")
    try:
        model = load_model(best_model_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    # 1. Initial Evaluation
    logger.debug("Initial evaluation on test set")
    test_metrics = model.evaluate(val_gen, steps=val_steps, return_dict=True)
    logger.info(f"Test Metrics: {test_metrics}")

    # 2. Collect predictions and true labels
    logger.debug("Collecting predictions and labels")
    y_true_list, y_score_list = [], []
    test_gen_iter = iter(val_gen)
    
    for _ in range(val_steps):
        x, y = next(test_gen_iter)
        y_true_list.append(y.numpy())
        y_score_list.append(model.predict(x, verbose=0))
    
    y_true = np.concatenate(y_true_list).ravel()
    y_score = np.concatenate(y_score_list).ravel()
    
    # 3. Threshold Optimization (if enabled)
    optimal_threshold = 0.5
    if threshold_tuning:
        logger.debug("Finding optimal threshold")
        from sklearn.metrics import f1_score
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = [f1_score(y_true, (y_score > t).astype(int)) for t in thresholds]
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (was 0.5)")

    y_pred = (y_score > optimal_threshold).astype(int)

    # 4. Comprehensive Metrics Calculation
    logger.debug("Calculating comprehensive metrics")
    from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                               f1_score, roc_auc_score, average_precision_score,
                               classification_report)
    
    metrics = {
        'accuracy': test_metrics['accuracy'],
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score),
        'pr_auc': average_precision_score(y_true, y_score),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }

    # 5. Save Metrics
    os.makedirs(metrics_output_dir, exist_ok=True)
    with open(os.path.join(metrics_output_dir, 'test_metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            if k not in ['confusion_matrix', 'classification_report']:
                f.write(f"{k}: {v:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(metrics['confusion_matrix']))
        f.write("\n\nClassification Report:\n")
        f.write(metrics['classification_report'])

    # 6. Enhanced Visualizations
    logger.debug("Generating visualizations")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    
    plt.figure(figsize=(15, 5))
    
    # ROC Curve
    plt.subplot(1, 3, 1)
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
    
    # Precision-Recall Curve
    plt.subplot(1, 3, 2)
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(f"PR Curve (AP = {metrics['pr_auc']:.2f})")
    
    # Threshold Analysis
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, f1_scores if threshold_tuning else [])
    plt.axvline(x=optimal_threshold, color='r', linestyle='--')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Optimization")
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_output_dir, 'test_metrics_plots.png'))
    plt.close()

    logger.info("Test evaluation complete. Metrics saved to %s", metrics_output_dir)
    return metrics
