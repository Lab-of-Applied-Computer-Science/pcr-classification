"""
    Module containing the nodes to download the data from the paper:
    {
        Duanmu et. al. A spatial attention guided deep learning system for prediction 
        of pathological complete response using breast cancer histopathology images.
        Bioinformatics, 38(19), 4605–4612. https://doi.org/10.1093/BIOINFORMATICS/BTAC558
    }
"""

import logging
import os

import boto3
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)

###########################
#    Helper Functions     #
###########################


def _get_s3_client():
    """Creates an AWS S3 client object.

    Returns:
        (botocore.client): S3 Client instantiated with local credentials.
    """
    logger.info("Retrieving S3 client with local credentials")
    conf_path = str(settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    credentials = conf_loader["credentials"]
    return boto3.client(
        "s3",
        aws_access_key_id=credentials["dev_s3"]["aws_access_key_id"],
        aws_secret_access_key=credentials["dev_s3"]["aws_secret_access_key"],
        region_name=credentials["dev_s3"]["region_name"],
    )


def _get_all_objects_from_root_folder(s3_client, bucket_name, root_folder):
    """Retrieves all objects from the specified bucket starting from the root folder.

    Args:
        s3_client (botocore.client): S3 Client instantiated with local credentials.
        bucket_name (str): Bucket from which to retrieve data.
        root_folder (str): Root folder name to download.
    """
    logger.info(
        "Listing all files inside bucket <%s> and root folder <%s>",
        bucket_name,
        root_folder,
    )
    is_resulted_truncated = True
    continuation_token = None

    obj_keys = []

    while is_resulted_truncated:
        if continuation_token is None:
            listed_objs = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=root_folder
            )
        else:
            listed_objs = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=root_folder,
                ContinuationToken=continuation_token,
            )

        for obj_content in listed_objs["Contents"]:
            obj_keys.append(obj_content["Key"])

        is_resulted_truncated = listed_objs["IsTruncated"]
        if is_resulted_truncated:
            continuation_token = listed_objs["NextContinuationToken"]

    return obj_keys


###########################
#         Nodes           #
###########################


def download_spatial_attention_pcr_data(
    bucket_name, root_folder, target_download_folder
):
    """Node that downloads the spatial attention PCR data.

    Args:
        bucket_name (str): Bucket from which to retrieve data.
        root_folder (str): Root folder name to download.
        target_download_folder (str): Target folder where to download the data.
    """
    s3_client = _get_s3_client()
    object_keys = _get_all_objects_from_root_folder(
        s3_client=s3_client, bucket_name=bucket_name, root_folder=root_folder
    )
    logger.info("Downloading files to %s", target_download_folder)
    for i, object_key in enumerate(object_keys):
        full_path = os.path.join(target_download_folder, object_key)

        if object_key.endswith("/"):
            continue

        logger.debug(
            "Downloading file (%d of %d): %s", i + 1, len(object_keys), object_key
        )

        if os.path.exists(full_path):
            logger.debug("File already exists! Skipping download...")
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            s3_client.download_file(
                Bucket=bucket_name,
                Key=object_key,
                Filename=full_path,
            )
