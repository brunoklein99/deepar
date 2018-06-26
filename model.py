import settings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(
            input_size=settings.INPUT_DIM,
            hidden_size=settings.HIDDEN_DIM,
            num_layers=3,
            batch_first=True
        )
        self.linear_m = nn.Linear(
            in_features=settings.HIDDEN_DIM,
            out_features=1
        )

        self.linear_a = nn.Linear(
            in_features=settings.HIDDEN_DIM,
            out_features=1
        )

    def forward(self, x):
        o, (_, _) = self.cell(x)
        m = F.softplus(self.linear_m(o))
        a = F.softplus(self.linear_a(o))
        return m, a
