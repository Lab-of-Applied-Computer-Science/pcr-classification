"""
    Module containing the nodes to download the data from the paper:
    {
        Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. 
        Structured crowdsourcing enables convolutional segmentation of histology images.
        Bioinformatics. 2019. doi: 10.1093/bioinformatics/btz083    
    }

"""
import os
import logging
from .download_crowdsource_dataset import main as run_data_downloader

logger = logging.getLogger(__name__)


def download_bcss_data(download_script_path):
    """
        Downloads the Breast Cancer Semantic Segmentation Dataset.
    """
    logger.info("Current working directory: %s", os.getcwd())
    run_data_downloader()
