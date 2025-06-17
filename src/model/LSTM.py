import torch
import torch.nn as nn

class singleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(singleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n[-1], c_n[-1]  # Return the last hidden and cell states