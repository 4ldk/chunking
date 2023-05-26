import torch
from transformers import AutoModelForTokenClassification, BertConfig

import preprocessing
import lstm
from datamodule import Net


batch_size = 1
lr = 0.0001
path = "./check_point/last-v2.ckpt"


def main():
    _, test_data, encode_dicts = preprocessing.preprocessing(chunk_pad_key="x")
    test_data, _, _ = preprocessing.subword_preprocessing(test_data, encode_dicts)

    chunk_dict = encode_dicts["chunk_dict"]

    device = "cuda"
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(chunk_dict))
    model = AutoModelForTokenClassification.from_config(config)
    model = lstm.dnn_crf(model, batch_size, len(chunk_dict))

    net = Net.load_from_checkpoint(
        model=model,
        lr=lr,
        crf=True,
        checkpoint_path=path,
    ).to(device)

    inputs, attention_mask, labels, out_texts, out_poses, out_chunks = (
        test_data["text"],
        test_data["attention_mask"],
        test_data["chunk"],
        test_data["raw_text"],
        test_data["raw_pos"],
        test_data["raw_chunk"],
    )
    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)

    output = []
    for input, mask, label, out_text, out_pos, out_chunk in zip(inputs, attention_mask, labels, out_texts, out_poses, out_chunks):

        input = input.unsqueeze(0)
        mask = mask.unsqueeze(0)
        pred_chunk = net.predict(input, mask).reshape(-1).to("cpu").tolist()
        pred_chunk = [preprocessing.val_to_key(p, chunk_dict) for (p, lbl) in zip(pred_chunk, label) if lbl != chunk_dict["x"]]
        pred_chunk = [c if c != "x" else "O" for c in pred_chunk]
        out = [" ".join([t, p, c, pred]) for t, p, c, pred in zip(out_text, out_pos, out_chunk, pred_chunk)]
        out = "\n".join(out)
        output.append(out)
        output.append("\n\n")

    output = "".join(output)

    with open("./output.txt", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
