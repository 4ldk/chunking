import torch
from torch import nn
from torch.nn import functional as F
from TorchCRF import CRF


class lstm(nn.Module):
    def __init__(self, batch_size, vocab_size, pos_size, chunk_dict, embedding_dim, hidden_dim, num_layers, device="cuda") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.chunk_dict = chunk_dict
        self.tagset_size = len(chunk_dict)
        self.num_layers = num_layers
        self.device = device

        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(pos_size, embedding_dim)
        self.ff1 = nn.Linear(embedding_dim * 2, embedding_dim * 4)
        self.ff2 = nn.Linear(embedding_dim * 4, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.output = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        inits = [
            torch.randn(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device),
            torch.randn(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device),
        ]
        return inits

    def forward(self, x, p):
        self.hidden = self.init_hidden()
        x = x.to(torch.int)
        x = torch.clamp(input=x, min=0, max=self.vocab_size - 1)
        p = p.to(torch.int)
        p = torch.clamp(input=p, min=0, max=self.pos_size - 1)

        word_embed = self.embeds(x)
        pos_embed = self.pos_embed(p)
        embeds = torch.concatenate([word_embed, pos_embed], dim=-1)

        embeds = F.relu(embeds)
        embeds = F.relu(self.ff1(embeds))
        embeds = F.relu(self.ff2(embeds))

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.output(lstm_out)

        return lstm_feats


class dnn_crf(nn.Module):
    def __init__(self, model, batch_size, num_labels, device="cuda") -> None:
        super().__init__()
        self.model = model
        self.crf = CRF(num_labels).to(device)
        self.mask = torch.ones(batch_size, 80).to(torch.bool).to(device)

    def forward(self, x, p, y):
        preds = self.model(x, p)
        if type(preds) == torch.Tensor:
            x = preds
        else:
            x = preds["logits"]
        out = self.crf.forward(x, y, self.mask)

        return out

    def decode(self, x, p):
        preds = self.model(x, p)
        if type(preds) == torch.Tensor:
            x = preds
        else:
            x = preds["logits"]
        out = self.crf.viterbi_decode(x, self.mask)
        out = torch.tensor(out)

        return out
