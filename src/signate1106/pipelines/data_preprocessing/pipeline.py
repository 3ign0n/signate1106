"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import enable_autologging, preprocess_train_data, preprocess_test_data, plot_image_size, check_label_num_bias, divide_image_folder_by_label

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=enable_autologging,
                inputs="parameters",
                outputs=None,
                name="enable_autologging_node",
            ),
            node(
                func=preprocess_train_data,
                inputs="train_data",
                outputs="preprocessed_train_data",
                name="preprocess_train_data_node",
            ),
            node(
                func=preprocess_test_data,
                inputs=None,
                outputs="preprocessed_test_data",
                name="preprocess_test_data_node",
            ),
            node(
                func=plot_image_size,
                inputs=["preprocessed_train_data", "preprocessed_test_data"],
                outputs=None,
                name="plot_image_size_node",
            ),
            node(
                func=check_label_num_bias,
                inputs=["preprocessed_train_data"],
                outputs=None,
                name="check_label_num_bias_node",
            ),
            node(
                func=divide_image_folder_by_label,
                inputs=["preprocessed_train_data", "preprocessed_test_data"],
                outputs=None,
                name="divide_image_folder_by_label_node",
            ),
    ])
