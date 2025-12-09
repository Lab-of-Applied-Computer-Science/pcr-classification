"""
This is a boilerplate pipeline 'train_cnn'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import train_cnn, split_dataset, test_cnn


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            # node(
            #     func=split_dataset,
            #     inputs=[
            #         "attention_map_tile_metadata",
            #         "params:mapping_split_train_val",
            #         "params:test_size",
                    
            #     ],
            #     outputs=["train_df","val_df"],
            #     name="split_dataset_node",
            # ),
            node(
                func=train_cnn,
                inputs=[
                    "train_df",
                    "val_df",
                    "params:num_classes",
                    "params:model_checkpoint_path",
                    "params:epochs",
                    "params:batch_size",
                    "params:learning_rate",
                    "params:weight_decay",
                    "params:metrics_output_dir",
                    "params:clearml_params",
                    "params:use_pretrained_weights"
                ],
                outputs=["val_gen", "val_steps"], # output val gen e val step
                name="train_pcr_classifier_node",
            ),
            node(
                func=test_cnn,
                inputs=[
                    "params:model_checkpoint_path",
                    "val_gen",
                    "val_steps",
                    "params:metrics_output_dir",
                    "params:threshold_tuning"
                ],
                outputs=None,
                name="test_pcr_classifier_node",
            ),
        ]
    )