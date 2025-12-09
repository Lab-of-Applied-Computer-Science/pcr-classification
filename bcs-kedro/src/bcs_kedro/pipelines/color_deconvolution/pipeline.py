"""Pipeline for applying color deconvolution"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_color_deconv


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_color_deconv,
                name="color_deconvolution",
                inputs=[
                    "params:src_dir",
                    "params:dst_dir",
                    "params:dab_threshold"
                ],
                outputs=None
            )
        ]
    )
