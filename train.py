import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import lstm
import preprocessing
from datamodule import lstm_dataset, Net

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
num_layers = 3
num_epoch = 500
batch_size = 50
lr = 0.0001


def main():
    train_data, test_data, encode_dicts = preprocessing.preprocessing()

    word_dict = encode_dicts["word_dict"]
    pos_dict = encode_dicts["pos_dict"]
    chunk_dict = encode_dicts["chunk_dict"]

    device = "cuda"
    model = lstm.lstm(batch_size, len(word_dict), len(pos_dict), chunk_dict, EMBEDDING_DIM, HIDDEN_DIM, num_layers=num_layers, device=device)
    model = lstm.dnn_crf(model, batch_size, len(chunk_dict), device=device)

    train_set = lstm_dataset(train_data["text"], train_data["pos"], train_data["chunk"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    test_set = lstm_dataset(test_data["text"], test_data["pos"], test_data["chunk"])
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print("input shape is")
    for x, p, y in train_loader:
        print(x.shape, p.shape, y.shape)
        break

    net = Net(model, lr, crf=True)

    callbacks = []
    checkpoint = ModelCheckpoint(
        dirpath="./check_point",
        filename="{epoch}-{f1:.2f}",
        monitor="f1",
        save_last=True,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )
    callbacks.append(checkpoint)
    """
    callbacks.append(
        EarlyStopping(
            "f1",
            patience=30,
            verbose=True,
            mode="max",
            check_on_train_epoch_end=False,
        )
    )
    """
    callbacks.append(
        EarlyStopping(
            "loss",
            patience=30,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        )
    )
    trainer = pl.Trainer(max_epochs=num_epoch, gpus=1, accelerator="gpu", check_val_every_n_epoch=10, callbacks=callbacks)

    trainer.fit(net, train_loader, test_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
