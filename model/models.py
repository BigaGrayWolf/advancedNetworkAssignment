import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=args.INPUT_SIZE,
                            hidden_size=args.HIDDEN_SIZE,
                            num_layers=args.LAYER_NUM,
                            batch_first=True)
        self.out = nn.Linear(args.HIDDEN_SIZE, 2)
    def forward(self, x):
        x = x.float()
        lstm_out, (h_n, c_n) = self.lstm(x)  # x (batch, time_step, input_size)
        out = self.out(lstm_out)
        return out