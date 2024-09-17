import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model.transforms import DEFAULT_TRANSFORM


class EyesDataset(Dataset):
    def __init__(self, data_DF, label_column, transform=DEFAULT_TRANSFORM):
        self.ann_df = data_DF
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return self.ann_df.shape[0]

    def __getitem__(self, index):
        row = self.ann_df.iloc[index]
        image = cv2.imread(row["img_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        label = None
        if self.label_column in self.ann_df.columns:
            label = row[self.label_column].astype(np.float32)
            label = torch.tensor(label)
        return image, label


class EyesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_csv_path,
        label_column,
        batch_size=100,
        train_transform=DEFAULT_TRANSFORM,
        test_transform=None,
        val_split=0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform if test_transform else train_transform
        data_df = pd.read_csv(data_csv_path)

        train_df, val_df = train_test_split(
            data_df,
            test_size=val_split,
            stratify=data_df[label_column],
            shuffle=True,
            random_state=42,
        )

        self.train_set = EyesDataset(
            train_df, label_column, transform=self.train_transform
        )
        self.val_set = EyesDataset(val_df, label_column, transform=self.test_transform)
        self.test_set = self.val_set

    def setup(self, stage=None):
        print(f"Train: {len(self.train_set)} images")
        print(f"Validation: {len(self.val_set)} images")
        print(f"Test: {len(self.test_set)} images")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
        )
