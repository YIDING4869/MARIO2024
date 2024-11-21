import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import OCTData
from evaluate import evaluate_model
from models import OCTClassifier
from training import train_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# class SquarePad:
#     def __call__(self, image):
#         max_wh = max(image.size)
#         pad_left = (max_wh - image.size[0]) // 2
#         pad_top = (max_wh - image.size[1]) // 2
#         pad_right = max_wh - image.size[0] - pad_left
#         pad_bottom = max_wh - image.size[1] - pad_top
#         return transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')


def main():
    set_seed(42)  # 你可以选择任何整数作为种子

    transform = transforms.Compose([
        # SquarePad(),  # 填充到正方形
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), # ??
    ])

    # dataset = OCTData.OCTData(csv_file='D:/AI_Data/data_1/df_task1_train_challenge.csv', root_dir='D:/AI_Data/data_1/train', transform=transform)
    dataset = OCTData.OCTData(csv_file='data/data_1/df_task1_train_challenge.csv',
                              root_dir='data/data_1/train', transform=transform)
    # 7:3 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

    model = OCTClassifier()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

    test_dataset = OCTData.TestData(csv_file='data/data_1/df_task1_val_challenge.csv',
                              root_dir='data/data_1/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    evaluate_model(model, test_loader, 'submission2.csv')

if __name__ == "__main__":
    main()



