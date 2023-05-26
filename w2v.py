from gensim.models import KeyedVectors
import torch
from torch import nn
import preprocessing
import lstm


def w2v_init_model(encode_dicts, batch_size, hidden_dim, num_layers, device="cuda"):

    word_dict = encode_dicts["word_dict"]
    pos_dict = encode_dicts["pos_dict"]
    chunk_dict = encode_dicts["chunk_dict"]
    wov_path = "D:/word_vector/GoogleNews-vectors-negative300.bin.gz"

    w2v = KeyedVectors.load_word2vec_format(wov_path, binary=True)

    embedding_dim = len(w2v["United_States"])
    embed_weight = []

    for key in word_dict.keys():
        if key in w2v.__dict__["index_to_key"]:
            embed_weight.append(w2v[key])
        else:
            init_weight = torch.rand(embedding_dim).tolist()
            embed_weight.append(init_weight)

    embed_weight = torch.tensor(embed_weight)

    model = lstm.lstm(batch_size, len(word_dict), len(pos_dict), chunk_dict, embedding_dim, hidden_dim, num_layers, device=device)

    model.embeds.Embedding = embed_weight

    return model
