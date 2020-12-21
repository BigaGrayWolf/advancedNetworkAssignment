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

class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=args.INPUT_SIZE,
                            hidden_size=args.HIDDEN_SIZE,
                            num_layers=args.LAYER_NUM,
                            batch_first=True)
        self.out = nn.Linear(args.HIDDEN_SIZE, 2)
    def forward(self, x):
        x = x.float()
        gru_out, h_n = self.gru(x)  # x (batch, time_step, input_size)
        out = self.out(gru_out)
        return out

class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=args.INPUT_SIZE,
                              hidden_size=args.HIDDEN_SIZE,
                              num_layers=args.LAYER_NUM,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(args.HIDDEN_SIZE*2, 2)
    def forward(self, x):
        x = x.float()
        bilstm_out, (h_n, c_n) = self.bilstm(x)  # x (batch, time_step, input_size)
        out = self.out(bilstm_out)
        return out