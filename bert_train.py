from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import numpy as np

import preprocessing
from sklearn.metrics import accuracy_score


class dataset(Dataset):
    def __init__(self, input, mask, label):
        self.features = [{"input": i, "mask": m, "label": lbl} for i, m, lbl in zip(input, mask, label)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return acc


def main():
    train_data, test_data, encode_dicts = preprocessing.preprocessing()
    train_data, encode_dicts, _ = preprocessing.subword_preprocessing(train_data, encode_dicts)
    test_data, _, _ = preprocessing.subword_preprocessing(test_data, encode_dicts)

    chunk_dict = encode_dicts["chunk_dict"]

    train_ids, train_mask = train_data["text"], train_data["attention_mask"]
    train_labels = train_data["chunk"]
    train_dataset = dataset(train_ids, train_mask, train_labels)

    test_ids, test_mask = test_data["text"], test_data["attention_mask"]
    test_labels = test_data["chunk"]
    test_dataset = dataset(test_ids, test_mask, test_labels)

    training_args = TrainingArguments(output_dir="test_trainer")
    model = AutoModelForTokenClassification("bert-base-cased", num_labels=len(chunk_dict))  # "dslim/bert-base-NER"

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
