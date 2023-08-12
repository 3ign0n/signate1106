"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.12
"""
import mlflow
from typing import Dict
from datetime import datetime
import pandas as pd
import os
from PIL import Image

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import shutil

def enable_autologging(parameters: Dict):
    #mlflow.set_tag("mlflow.runName", datetime.now().isoformat())
    #mlflow.autolog()
    print("skipping enable_autologging func for now")

def __add_image_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    width_list, height_list, ratio_list = [], [], []
    for _, row in df.iterrows():
        img = Image.open(row['path'])
        width, height = img.size
        width_list.append(width)
        height_list.append(height)
        ratio_list.append(width / height)

    df.insert(2, column='aspect ratio', value=ratio_list)
    df.insert(2, column='height', value=height_list)
    df.insert(2, column='width', value=width_list)
    return df

def preprocess_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    train_data['path'] = os.getcwd() + '/data/01_raw/train/' + train_data['image_name']
    train_data = train_data.reindex(columns=['image_name', 'path', 'label'])

    train_data = __add_image_size_columns(train_data)

    return train_data

def preprocess_test_data() -> pd.DataFrame:
    data_dir=os.getcwd() + '/data/01_raw/test/'

    file_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    df = pd.DataFrame(file_list, columns=['image_name'])
    df['path'] = data_dir + df['image_name']

    df = __add_image_size_columns(df)

    return df

def plot_image_size(preprocessed_train_data: pd.DataFrame, preprocessed_test_data: pd.DataFrame):
    fig = make_subplots(rows=2, cols=3, column_titles=['width', 'height', 'aspect ratio'], row_titles=['train', 'test'])

    def __plot_image_size_at_row(df: pd.DataFrame, fig: go.Figure, row: int):
        fig.add_trace(
            go.Histogram(x=df['width'], xbins=dict(start=400, end=600, size=50)),
            row=row, col=1
        )
        fig.update_xaxes(range=[400, 600], row=row, col=1)

        fig.add_trace(
            go.Histogram(x=df['height'], xbins=dict(start=400, end=600, size=50)),
            row=row, col=2
        )
        fig.update_xaxes(range=[400, 600], row=row, col=2)

        fig.add_trace(
            go.Histogram(x=df['aspect ratio'], nbinsx=6),
            row=row, col=3
        )
        fig.update_xaxes(range=[0, 3], row=row, col=3)

    __plot_image_size_at_row(preprocessed_train_data, fig, 1)
    __plot_image_size_at_row(preprocessed_test_data, fig, 2)

    output_dir='data/08_reporting/image_size'
    os.makedirs(output_dir, exist_ok=True)

    # write_imageは、上書きオプションないようなので、ファイルが存在していたら削除する
    image_file=os.path.join(output_dir, 'image_size.png')
    os.remove(image_file) if os.path.exists(image_file) else None
    fig.write_image(image_file)

def check_label_num_bias(preprocessed_train_data: pd.DataFrame):
    """
    labelに、0:飲料と1:食料の数に偏りがあるか確認
    """
    df = preprocessed_train_data
    drink_cnt = len(df[df['label'] == 0])
    food_cnt = len(df[df['label'] == 0])
    print(f"drink_cnt:{drink_cnt}, food_cnt:{food_cnt}")

def divide_image_folder_by_label(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    後でtorchvision.ImageFolderを使うため、labelに応じてディレクトリを分けて画像を格納
    """

    output_dir='data/02_intermediate/classified_train_images'
    labels = train_data['label'].unique()
    for label in labels:
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)

    for _, row in train_data.iterrows():
        shutil.copy2(row['path'], os.path.join(output_dir, str(row['label'])))

    # 評価用データの方は、ラベルがわからないので一旦全部0:飲料のディレクトリにまとめておく
    output_dir='data/02_intermediate/classified_test_images/0'
    os.makedirs(output_dir, exist_ok=True)
    for _, row in test_data.iterrows():
        shutil.copy2(row['path'], output_dir)

