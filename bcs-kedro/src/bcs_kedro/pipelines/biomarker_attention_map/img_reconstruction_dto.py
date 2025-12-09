from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ImageReconstructionDTO:
    img_ki67: np.ndarray
    img_phh3: np.ndarray
    img_seg: np.ndarray
    output_image_path: str
    metadata: pd.DataFrame