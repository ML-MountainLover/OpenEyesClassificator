import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve
from torch.nn import functional as F


class EyesClassifier(pl.LightningModule):
    def __init__(self, freeze=None):
        super().__init__()

        self.face_model = InceptionResnetV1(pretrained="vggface2", classify=True)
        linear_size = self.face_model.logits.in_features
        self.face_model.logits = nn.Sequential(nn.Linear(linear_size, 2, bias=True))
        for child in list(self.face_model.children()):
            for param in child.parameters():
                param.requires_grad = True

        if freeze == "last":
            for child in list(self.face_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == "most":
            for child in list(self.face_model.children())[:-5]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze is not None:
            raise NotImplementedError("Wrong freezing parameter")

    def forward(self, x):
        return self.face_model(x)


class LightningEyesClassifier(pl.LightningModule):
    def __init__(self, lr_rate=5e-4, freeze=None):
        super(LightningEyesClassifier, self).__init__()

        self.model = EyesClassifier(freeze)
        self.lr_rate = lr_rate
        self.train_loss = []
        self.train_eer = []
        self.val_loss = []
        self.val_eer = []

    def forward(self, x):
        return self.model(x)

    def EER_score(self, logits, gt):
        logits = logits.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        if len(np.unique(gt)) < 2:
            return torch.tensor(-1.0)
        fpr, tpr, _ = roc_curve(gt, logits, pos_label=1)
        fnr = 1 - tpr
        if np.isnan(fpr).any() or np.isnan(fnr).any():
            return torch.tensor(-1.0)
        EER = fpr[np.nanargmin(np.abs(fnr - fpr))]
        return torch.tensor(EER)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.train_loss.append(loss.detach().cpu())
        logs = {"train_loss": loss}
        eer = self.EER_score(torch.softmax(logits, dim=1)[:, 1].detach().cpu(), y)
        self.train_eer.append(eer.detach().cpu())
        self.log("train_eer", eer, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.val_loss.append(loss.detach().cpu())
        eer = self.EER_score(torch.softmax(logits, dim=1)[:, 1].detach().cpu(), y)
        self.val_eer.append(eer.detach().cpu())
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_eer": eer}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        eer = self.EER_score(torch.softmax(logits, dim=1)[:, 1].detach().cpu(), y)
        self.log("test_eer", eer, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"test_loss": loss, "test_eer": eer}

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_loss).mean()
        avg_eer = torch.stack(self.train_eer).mean()
        print(
            f"\nEpoch {self.trainer.current_epoch}, "
            f"Train_loss: {round(float(avg_loss), 5)} "
            f"Train_eer: {round(float(avg_eer), 5)}\n",
        )
        self.train_loss.clear()
        self.train_eer.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_eer = torch.stack(self.val_eer).mean()
        print(
            f"\nEpoch {self.trainer.current_epoch}, "
            f"Val_loss: {round(float(avg_loss), 5)} "
            f"Val_eer: {round(float(avg_eer), 5)}\n",
        )
        self.val_loss.clear()
        self.val_eer.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr_rate, weight_decay=1e-4
        )
        return [optimizer]

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr_rate, weight_decay=1e-2
        )
        self.reduce_lr_on_plateau = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.5, patience=1, min_lr=1e-7, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
        }
        return [self.optimizer], [self.reduce_lr_on_plateau]
