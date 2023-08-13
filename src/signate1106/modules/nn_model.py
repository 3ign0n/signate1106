import pandas as pd
import os
import copy
from typing import Dict, Any

import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score

from PIL import ImageMode

from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F

# Google CollabでTPUを使うためのimport
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    pass


"""
# カスタムデータセット
class ImgDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    #dataframeから画像へのパスとラベルを読み出す
    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        image = io.imread(self.df.iloc[idx]['path'])
        image = self.transform(image) if self.transform else image
        return image, label
"""
    
"""
class ImgNet(nn.Module):

    def __init__(self):
        super().__init__() 

        self.c0 = nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2, 
                            padding=1)

        self.c1 = nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1)

        self.c2 = nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1)   

        self.bn0 = nn.BatchNorm2d(num_features=16)   # c0用のバッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=32)   # c1用のバッチ正則化
        self.bn2 = nn.BatchNorm2d(num_features=64)   # c2用のバッチ正則化

        self.fc = nn.Linear(in_features=64 * 28 * 28,   # 入力サイズ
                            out_features=2)

    def __call__(self, x): 
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))  
        h = h.view(-1, 64 * 28 * 28)
        y = self.fc(h)     # 全結合層
        return y
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_nn_model(train_df: pd.DataFrame, parameters: Dict) -> Any:
    model_opts=parameters["model_opts"]
    __set_random_state(parameters["random_state"])

    phases=data_transforms.keys()

    model_ft = models.resnet18(pretrained=True)
    n_ftrs = model_ft.fc.in_features

    # 0:飲料と1:食料の2クラス
    model_ft.fc = nn.Linear(n_ftrs, 2)
    #print("default requires_grad parameters:")
    #for name, param in model_ft.named_parameters():
    #    print(f"Name: {name}, requires_grad: {param.requires_grad}")

    for param in model_ft.parameters():
        param.requires_grad=model_opts['requires_grad']

    print("changed requires_grad parameters:")
    for name, param in model_ft.named_parameters():
        print(f"Name: {name}, requires_grad: {param.requires_grad}")

    if os.environ['COLAB_TPU_ADDR']:
        device = xm.xla_device()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device:{device}")
    model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_best=[]
    if model_opts["n_folds"] == 1:
        img_datasets = {x: datasets.ImageFolder('data/02_intermediate/classified_train_images', transform=data_transforms[x]) for x in phases}
        model = __train_model(img_datasets, model_opts, phases, device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=model_opts['n_epochs'])
        model_best = model_best.append(model)

    else:
        image_dataset = datasets.ImageFolder('data/02_intermediate/classified_train_images')
        kf = KFold(n_splits=model_opts["n_folds"], shuffle=True, random_state=parameters["random_state"])
        for _fold, (train_index, valid_index) in enumerate(kf.split(image_dataset)):
            print(f"======={_fold}")

            img_datasets = {x: Subset(image_dataset, train_index, transforms=data_transforms[x]) for x in phases}

            model = __train_model(img_datasets, model_opts, phases, device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=model_opts['n_epochs'])
            model_best = model_best.append(model)
            #__visualize_model(model_ft)

    return model_best

def __set_random_state(random_state: int):
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True    

def __train_model(img_datasets, model_opts, phases, device, model, criterion, optimizer, scheduler, num_epochs):

    dataloaders = {x: DataLoader(img_datasets[x], batch_size=model_opts['batch_size'], shuffle=True, num_workers=4) for x in phases}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            epoch_labels = np.empty(0)
            epoch_preds = np.empty(0)
            for inputs, labels in dataloaders[phase]:
                epoch_labels = np.append(epoch_labels, labels.numpy())

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    epoch_preds = np.append(epoch_preds, preds.cpu().detach().numpy())

                    # result type Float can't be cast to the desired output type Long になる
                    # BCEWithLogitsLossはfloatを要求するから？
                    loss = criterion(preds.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if model_opts['requires_grad']:
                            loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(img_datasets[phase])
            epoch_acc = running_corrects.double() / len(img_datasets[phase])
            epoch_sk_acc = accuracy_score(epoch_labels, epoch_preds)
            epoch_auc = roc_auc_score(epoch_labels, epoch_preds)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, SK Acc: {epoch_sk_acc}, SK AUC: {epoch_auc}')

            # deep copy the model
            if phase == 'valid' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

                # FIXME: ここは後で直す
                torch.save(best_model_wts, f"data/06_models/best_model_wts.pth")

    print(f'Best val Acc: {best_auc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


"""
def __visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
"""


def predict(model, test_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    model_opts=parameters['model_opts']

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_dataset = datasets.ImageFolder('data/02_intermediate/classified_test_images', transform=data_transforms['valid'])
    dataloader = DataLoader(img_dataset, batch_size=model_opts['batch_size'], shuffle=True, num_workers=4)


    predict_list = []
    for images, _ in dataloader:
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            predicts = outputs.softmax(dim=1)

            predicts = predicts.cpu().detach().numpy()

            predict_list = np.append(predict_list, predicts[:,1])

    #予測値が1である確率を提出します。
    output_df = pd.read_csv('data/01_raw/sample_submit.csv', header=None)
    output_df[1] = predict_list

    return output_df