"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=train_model,
                inputs=["preprocessed_train_data", "parameters"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=predict,
                inputs=["preprocessed_test_data", "parameters"],
                outputs="y_pred",
                name="predict_node",
            ),
    ])
