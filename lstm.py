import torch
from torch import nn
from TorchCRF import CRF

START_TAG = "SOS"
STOP_TAG = "EOS"


class lstm(nn.Module):
    def __init__(self, batch_size, vocab_size, chunk_dict, embedding_dim, hidden_dim, num_layers, device="cuda") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.chunk_dict = chunk_dict
        self.tagset_size = len(chunk_dict)
        self.num_layers = num_layers
        self.device = device

        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.output = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        inits = [
            torch.randn(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device),
            torch.randn(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device),
        ]
        return inits

    def forward(self, x):
        self.hidden = self.init_hidden()
        x = x.to(torch.int)
        x = torch.clamp(input=x, min=0, max=self.vocab_size - 1)
        embeds = self.embeds(x)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.output(lstm_out)

        return lstm_feats


class dnn_crf(nn.Module):
    def __init__(self, model, batch_size, num_labels, device="cuda") -> None:
        super().__init__()
        self.model = model
        self.crf = CRF(num_labels).to(device)
        self.mask = torch.ones(batch_size, 80).to(torch.bool).to(device)

    def forward(self, x, y):
        x = self.model(x)
        out = self.crf.forward(x, y, self.mask)

        return out

    def decode(self, x):
        x = self.model(x)
        out = self.crf.viterbi_decode(x, self.mask)
        return out
