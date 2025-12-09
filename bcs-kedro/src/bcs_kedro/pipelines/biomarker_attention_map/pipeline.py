"""
    Pipeline for generating an attention map from IHC and H&E images.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_attention_maps


def create_pipeline(**kwargs) -> Pipeline:
    """
        Creation of the pipeline for generating the attention map for IHC and segmented images.

    Returns:
        (Pipeline): Returns the pipeline with all instantiated nodes.
    """
    return pipeline(
        [
            node(
                func=generate_attention_maps,
                name="biomarker_attention_map",
                inputs=[
                    "params:src_dir_deconv_tiles",
                    "params:src_dir_segmented_tiles",
                    "params:dst_dir_attention_maps",
                    "params:tile_width",
                    "params:tile_height",
                    "params:orig_img_width",
                    "params:orig_img_height",
                    "params:bandwidth",
                    "params:kde_samples",
                    "params:use_tumor_cell",
                ],
                outputs=None
            )
        ]
    )
