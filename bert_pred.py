from transformers import AutoModelForTokenClassification

import preprocessing

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
num_layers = 3
num_epoch = 500
batch_size = 1
lr = 0.0001
path = "./bert_model"


def main():
    _, test_data, encode_dicts = preprocessing.preprocessing(chunk_pad_key="x")
    test_data, _, _ = preprocessing.subword_preprocessing(test_data, encode_dicts)

    chunk_dict = encode_dicts["chunk_dict"]

    model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    model = model.to("cuda")
    model = model.eval()

    output = []
    for input, mask, label, out_text, out_pos, out_chunk in zip(
        test_data["text"], test_data["attention_mask"], test_data["chunk"], test_data["raw_text"], test_data["raw_pos"], test_data["raw_chunk"]
    ):

        input = input.squeeze(0).to("cuda")
        mask = mask.squeeze(0).to("cuda")
        pred_chunk = model(input, mask).to("cpu").argmax(axis=-1).reshape(-1).tolist()
        pred_chunk = [preprocessing.val_to_key(p, chunk_dict) for (p, lbl) in zip(pred_chunk, label) if lbl != len(chunk_dict)]
        out = [" ".join([t, p, c, pred]) for t, p, c, pred in zip(out_text, out_pos, out_chunk, pred_chunk)]
        out = "\n".join(out)
        output.append(out)
        output.append("\n\n")

    output = "".join(output)

    with open("./output.txt", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
