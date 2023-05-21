import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

import preprocessing
from datamodule import Net, bert_dataset
import lstm


batch_size = 25
lr = 1e-4
num_epoch = 50


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return acc


def main():
    train_data, test_data, encode_dicts = preprocessing.preprocessing(chunk_pad_key="x")
    train_data, encode_dicts, _ = preprocessing.subword_preprocessing(train_data, encode_dicts)
    test_data, _, _ = preprocessing.subword_preprocessing(test_data, encode_dicts)

    chunk_dict = encode_dicts["chunk_dict"]

    train_ids, train_mask = train_data["text"], train_data["attention_mask"]
    train_labels = train_data["chunk"]
    train_set = bert_dataset(train_ids, train_mask, train_labels)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    test_ids, test_mask = test_data["text"], test_data["attention_mask"]
    test_labels = test_data["chunk"]
    test_set = bert_dataset(test_ids, test_mask, test_labels)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(chunk_dict))  # "dslim/bert-base-NER"
    model = lstm.dnn_crf(model, batch_size, len(chunk_dict))

    net = Net(model, lr, crf=True)

    callbacks = []
    checkpoint = ModelCheckpoint(
        dirpath="./check_point",
        filename="{epoch}-{recall:.2f}",
        monitor="acc",
        save_last=True,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )
    callbacks.append(checkpoint)

    trainer = pl.Trainer(max_epochs=num_epoch, gpus=1, accelerator="gpu", check_val_every_n_epoch=10, callbacks=callbacks)

    trainer.fit(net, train_loader, test_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
