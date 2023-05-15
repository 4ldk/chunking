import torch

import lstm
import preprocessing
from datamodule import Dataset, Net

START_TAG = lstm.START_TAG
STOP_TAG = lstm.STOP_TAG

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
num_layers = 3
num_epoch = 500
batch_size = 1
lr = 0.0001
path = "./check_point/last.ckpt"


def main():
    _, test_data, encode_dicts = preprocessing.preprocessing()

    word_dict = encode_dicts["word_dict"]
    chunk_dict = encode_dicts["chunk_dict"]
    chunk_dict[START_TAG] = len(chunk_dict.keys())
    chunk_dict[STOP_TAG] = len(chunk_dict.keys())

    model = lstm.lstm(batch_size, len(word_dict), chunk_dict, EMBEDDING_DIM, HIDDEN_DIM, num_layers=num_layers)
    model = lstm.dnn_crf(model, batch_size, len(chunk_dict))

    net = Net.load_from_checkpoint(
        model=model,
        lr=lr,
        crf=True,
        checkpoint_path=path,
    ).to("cuda")

    output = []
    for input, label, out_text, out_pos, out_chunk in zip(
        test_data["text"], test_data["chunk"], test_data["raw_text"], test_data["raw_pos"], test_data["raw_chunk"]
    ):
        input = torch.tensor([input]).to("cuda")
        pred_chunk = net.predict(input).to("cpu").reshape(-1).tolist()
        pred_chunk = [preprocessing.val_to_key(p, chunk_dict) for (p, lbl) in zip(pred_chunk, label) if lbl != len(chunk_dict)]
        pred_chunk = [c for c in pred_chunk if c != "PAD"]
        out = [" ".join([t, p, c, pred]) for t, p, c, pred in zip(out_text, out_pos, out_chunk, pred_chunk)]
        out = "\n".join(out)
        output.append(out)
        output.append("\n\n")

    output = "".join(output)

    with open("./output.txt", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
