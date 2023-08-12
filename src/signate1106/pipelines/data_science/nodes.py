"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.12
"""
from typing import Dict, Any
import pandas as pd
import signate1106.modules.nn_model as nn_model

def train_model(df: pd.DataFrame, parameters: Dict) -> Any:
    return nn_model.train_nn_model(df, parameters)

import torch
from torchvision import models

def predict(test_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("data/06_models/best_model_wts.pth"))
    return nn_model.predict(model, test_data, parameters)
