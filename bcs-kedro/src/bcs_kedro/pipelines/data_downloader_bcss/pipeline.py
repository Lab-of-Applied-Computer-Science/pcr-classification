"""
    Pipeline for downloading the BCSS dataset.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_bcss_data

def create_pipeline(**kwargs) -> Pipeline:
    """
        Creation of the pipeline for downloading the BCSS dataset.

    Returns:
        (Pipeline): Returns the pipeline with all instantiated nodes.
    """
    return pipeline(
        [
            node(
                func=download_bcss_data,
                name="download_bcss_data",
                inputs=["params:download_script_path"],
                outputs=None,
            )
        ]
    )
