import re
from itertools import chain
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer


def key_to_val(key, dic):

    return dic[key] if key in dic else dic["UNK"]


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys == pad_key:
        return "PAD"
    elif keys:
        return keys[0]
    return "UNK"


def encode(data, encode_dict, pad_key="PAD", bos_key="BOS", eos_key="EOS"):
    out = []
    for d in data:
        val_list = [encode_dict[bos_key]]
        val_list += [key_to_val(key, encode_dict) for key in d]

        while len(val_list) < 80 - 1:
            val_list.append(key_to_val(pad_key, encode_dict))

        val_list += [encode_dict[eos_key]]

        out.append(val_list)
    return out


def decode(data, encode_dict):
    out = []
    for d in data:
        out.append([val_to_key(val, encode_dict) for val in d])

    return out


def make_dict(data, min_num=0, pad_key="PAD", bos_key="BOS", eos_key="EOS"):
    keys = Counter(list(chain.from_iterable(data)))
    keys = (k for (k, v) in keys.items() if v > min_num)
    d = {k: i for i, k in enumerate(keys)}

    d["UNK"] = len(d)
    d[pad_key] = len(d)
    d[bos_key] = len(d)
    d[eos_key] = len(d)

    return d


def num_masking(t):
    num_count = sum([int(chr.isdigit()) for chr in t])
    if num_count / len(t) > 0.75:
        return "NUM"
    else:
        return t


def simplify_text(text):
    simple_text = []
    for word in text:
        simple_word = [num_masking(w.lower()) for w in word]
        simple_text.append(simple_word)

    return simple_text


def path_to_data(path, encode_dicts={}):

    with open(path, "r") as f:
        data = f.read()
    # data = re.sub("0-9", "0", data.lower())
    data = data.split("\n\n")
    data = [d.split("\n") for d in data]

    text = []
    pos = []
    chunk = []

    for sentence in tqdm(data):
        if len(sentence) < 5:
            continue

        divided = [s.split(" ") for s in sentence]
        text.append([d[0] for d in divided])
        pos.append([d[1] for d in divided])
        chunk.append([d[2] for d in divided])
        # chunk = simplify_chunk(chunk)

    simple_text = simplify_text(text)

    if "word_dict" not in encode_dicts.keys():
        word_dict = make_dict(simple_text, min_num=2)
    else:
        word_dict = encode_dicts["word_dict"]

    if "pos_dict" not in encode_dicts.keys():
        pos_dict = make_dict(pos)
    else:
        pos_dict = encode_dicts["pos_dict"]

    if "chunk_dict" not in encode_dicts.keys():
        chunk_dict = make_dict(chunk)
    else:
        chunk_dict = encode_dicts["chunk_dict"]

    e_text = encode(simple_text, word_dict)
    e_pos = encode(pos, pos_dict)
    e_chunk = encode(chunk, chunk_dict)

    data = {
        "text": e_text,
        "pos": e_pos,
        "chunk": e_chunk,
        "raw_text": text,
        "raw_pos": pos,
        "raw_chunk": chunk,
    }
    encode_dicts = {
        "word_dict": word_dict,
        "pos_dict": pos_dict,
        "chunk_dict": chunk_dict,
    }

    return data, encode_dicts


def preprocessing():
    dataset = "D:/download/conll2000"

    train = dataset + "/train.txt"
    test = dataset + "/test.txt"

    train_data, encode_dicts = path_to_data(train)
    test_data, _ = path_to_data(test, encode_dicts)

    return train_data, test_data, encode_dicts


def subword_preprocessing(data, encode_dicts):
    e_pos, e_chunk, raw_text, raw_pos, raw_chunk = (
        data["pos"],
        data["chunk"],
        data["raw_text"],
        data["raw_pos"],
        data["raw_chunk"],
    )
    chunk_dict = encode_dicts["chunk_dict"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # "dslim/bert-base-NER"

    tokens = tokenizer(raw_text, truncation=True, is_split_into_words=True, return_tensors="pt", max_length=80, padding="max_length")

    labels = []
    for i, label in enumerate(e_chunk):
        word_ids = tokens.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(chunk_dict["x"])
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(chunk_dict["x"])
            previous_word_idx = word_idx
        labels.append(label_ids)

    print(tokens["input_ids"][0])
    print(labels[0])

    print(tokens["input_ids"][1])
    print(labels[1])

    print(tokens["input_ids"][2])
    print(labels[2])

    data = {
        "text": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "pos": e_pos,
        "chunk": labels,
        "raw_text": raw_text,
        "raw_pos": raw_pos,
        "raw_chunk": raw_chunk,
    }
    return data, encode_dicts, tokenizer


if __name__ == "__main__":
    train_data, test_data, encode_dicts = preprocessing()
    # train_data, encode_dicts, tokenizer = subword_preprocessing(train_data, encode_dicts)
    # test_data, _, _ = subword_preprocessing(test_data, encode_dicts)

    print(encode_dicts["chunk_dict"])

    text = decode(train_data["text"][0:1], encode_dicts["word_dict"])[0]
    print(text, len(text))
    # for t in train_data["chunk"][:10]:
    #    print(t)
