"""
    Pipeline for downloading the spatial attention PCR dataset.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_spatial_attention_pcr_data


def create_pipeline(**kwargs) -> Pipeline:
    """
        Creation of the pipeline for downloading the spatial attention PCR dataset.

    Returns:
        (Pipeline): Returns the pipeline with all instantiated nodes.
    """
    return pipeline(
        [
            node(
                func=download_spatial_attention_pcr_data,
                name="download_spatial_attention_pcr_data",
                inputs=[
                    "params:bucket_name",
                    "params:root_folder",
                    "params:target_download_folder",
                ],
                outputs=None,
            )
        ]
    )
