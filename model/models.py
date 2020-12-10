import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=LAYER_NUM,
                            batch_first=True)
        self.out = nn.Linear(HIDDEN_SIZE, 2)
    def forward(self, x):
        x = x.float()
        r_out, (h_n, h_c) = self.lstm(x, None)  # x (batch, time_step, input_size)
        out = self.out(r_out)
        return out