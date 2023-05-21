import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch import nn, optim

from utils import recall_score, f1_score


class Dataset(Dataset):
    def __init__(self, X, P, y) -> None:
        super().__init__()
        self.X = torch.tensor(X)
        self.P = torch.tensor(P)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.P[idx], self.y[idx]


class bert_dataset(Dataset):
    def __init__(self, X, mask, y) -> None:
        super().__init__()
        if type(X) != torch.Tensor:
            X = torch.tensor(X)
        if type(mask) != torch.Tensor:
            mask = torch.tensor(mask)
        if type(y) != torch.Tensor:
            y = torch.tensor(y)

        self.X = X
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


class Net(pl.LightningModule):
    def __init__(self, model, lr, crf=False):
        super().__init__()

        print("Making Model...")
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.crf = crf

    def forward(self, x, sub):
        output = self.model(x, sub)
        return output

    def predict(self, x, sub):
        if self.crf:
            output = self.model.decode(x, sub)
            output = torch.tensor(output)
        else:
            output = self.forward(x)
        return output

    def loss_fn(self, pred, label):

        pred = pred.reshape(-1, pred.shape[-1])
        label = label.view(-1).to(torch.int64)
        loss = self.criterion(pred, label)

        return loss

    def training_step(self, batch, batch_idx):

        input, sub_input, label = batch

        if self.crf:
            loss = self.model.forward(input, sub_input, label)
            loss = torch.sum(-loss)
        else:
            pred = self.forward(input, sub_input)
            loss = self.loss_fn(pred, label)
        self.log("loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        input, sub_input, label = batch
        batch_size = input.shape[0]
        label = label.to("cpu")
        pred = self.predict(input, sub_input).to("cpu")
        pred = pred.squeeze(-1)
        acc = (pred == label).sum().item() / batch_size
        pred = pred.view(-1)
        label = label.view(-1)
        recall = recall_score(label, pred, average="macro").tolist()

        return {"acc": acc, "recall": recall}

    def validation_epoch_end(self, outputs):
        ave_acc = torch.tensor([x["acc"] for x in outputs]).to(torch.float).mean()
        ave_recall = torch.tensor([x["recall"] for x in outputs]).to(torch.float).mean()

        self.log("acc", ave_acc)
        self.log("recall", ave_recall)
        self.log("lr", self.optimizer.param_groups[0]["lr"])

        return {"acc": ave_acc}

    def test_step(self, batch, batch_idx):

        input, sub_input, label = batch
        batch_size = input.shape[0]
        label = label.to("cpu")
        pred = self.predict(input, sub_input).to("cpu")

        pred = pred.squeeze(-1)
        acc = (pred == label).sum().item() / batch_size
        pred = pred.view(-1)
        label = label.view(-1)
        recall = recall_score(label, pred, average="macro").tolist()
        f1 = f1_score(label, pred).tolist()

        self.log("test_acc", acc)
        self.log("test_recall", recall)
        self.log("test_f1", f1)

    def test_epoch_end(self, outputs) -> None:
        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        return [self.optimizer]
