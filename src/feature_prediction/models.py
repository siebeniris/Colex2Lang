import torch
from torch import nn
import warnings

warnings.filterwarnings("ignore")


class OneFF(nn.Module):
    """
    Only one feedforward layer, followed by linear layer for classification
    Initialize embeddings for languages randomly with uniform distribution.
    """

    def __init__(self, device, num_langs, input_dim, hidden_dim, label_dim, dropout):
        super(OneFF, self).__init__()
        self.label_dim = label_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, self.label_dim)
        self.device = device
        self.num_langs = num_langs
        self.input_dim = input_dim

        self.fc.weight.data.uniform_(-1.0, 1.0)
        self.fc.bias.data.zero_()

    def forward(self, input_idx, language_vector=None):

        if language_vector is not None:
            input_embeddings = torch.tensor(language_vector)
            input_embeddings = input_embeddings.to(self.device)

        else:
            # initialize ones rather than uniform distribution.
            emb1 = nn.Embedding(self.num_langs, self.input_dim)
            emb1 = nn.init.ones_(emb1.weight)
            input_embeddings = emb1[input_idx]

        fc_output = self.fc(input_embeddings)
        droput_layer = self.dropout_layer(fc_output)
        output = self.classifier(droput_layer)

        return output
