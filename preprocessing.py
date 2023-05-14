import re
from itertools import chain

from tqdm import tqdm


def key_to_val(key, dic):
    return dic[key] if key in dic else len(dic)


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys == pad_key:
        return "PAD"
    elif keys:
        return keys[0]
    return "UNK"


def encode(data, encode_dict, pad_key="PAD"):
    out = []
    for d in data:
        val_list = [key_to_val(key, encode_dict) for key in d]
        while len(val_list) < 80:
            val_list.append(key_to_val(pad_key, encode_dict))

        out.append(val_list)
    return out


def decode(data, encode_dict):
    out = []
    for d in data:
        out.append([val_to_key(val, encode_dict) for val in d])

    return out


def make_dict(data):
    keys = list(set(chain.from_iterable(data)))
    keys.sort()
    d = {k: i for i, k in enumerate(keys)}

    return d


def simplify_chunk(chunk):
    out = []
    for c in chunk:
        simple_chunk = ["i" if val[0] == "i" else val for val in c]
        simple_chunk = ["b" if val[0] == "b" else val for val in c]
        out.append(simple_chunk)

    return out


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

    if "word_dict" not in encode_dicts.keys():
        word_dict = make_dict(text)
        word_dict["PAD"] = len(word_dict)
    else:
        word_dict = encode_dicts["word_dict"]

    if "pos_dict" not in encode_dicts.keys():
        pos_dict = make_dict(pos)
        pos_dict["PAD"] = len(pos_dict)
    else:
        pos_dict = encode_dicts["pos_dict"]

    if "chunk_dict" not in encode_dicts.keys():
        chunk_dict = make_dict(chunk)
        pos_dict["PAD"] = len(pos_dict)
    else:
        chunk_dict = encode_dicts["chunk_dict"]

    e_text = encode(text, word_dict)
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


if __name__ == "__main__":
    train_data, test_data, encode_dicts = preprocessing()

    print(encode_dicts["chunk_dict"])
    for t in train_data["chunk"][:10]:
        print(t)