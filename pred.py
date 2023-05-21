import torch

import lstm
import preprocessing
from datamodule import Dataset, Net


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
    pos_dict = encode_dicts["pos_dict"]
    chunk_dict = encode_dicts["chunk_dict"]

    device = "cpu"
    model = lstm.lstm(batch_size, len(word_dict), len(pos_dict), chunk_dict, EMBEDDING_DIM, HIDDEN_DIM, num_layers=num_layers, device=device)
    model = lstm.dnn_crf(model, batch_size, len(chunk_dict), device=device)

    net = Net.load_from_checkpoint(
        model=model,
        lr=lr,
        crf=True,
        checkpoint_path=path,
    )

    output = []
    for input, pos, out_text, out_pos, out_chunk in zip(
        test_data["text"], test_data["pos"], test_data["raw_text"], test_data["raw_pos"], test_data["raw_chunk"]
    ):
        input = torch.tensor([input])
        pos = torch.tensor([pos])
        pred_chunk = net.predict(input, pos).reshape(-1).tolist()
        pred_chunk = pred_chunk[1:-1]
        pred_chunk = [preprocessing.val_to_key(p, chunk_dict) for p in pred_chunk]
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
