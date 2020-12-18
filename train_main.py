import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.models import *


def args_parse():
    parser = argparse.ArgumentParser(description='abnormal flow detection')
    parser.add_argument('--MODEL', type=str, help='type of RNN model (by default LSTM)')
    parser.add_argument('--EPOCH', type=int, help='number of epochs')
    parser.add_argument('--BATCH-SIZE', type=int, help='minibatch size')
    parser.add_argument('--TIME-STEP', type=int, help='number of time steps')
    parser.add_argument('--INPUT-SIZE', type=int, help='dimension of input')
    parser.add_argument('--HIDDEN-SIZE', type=int, help='dimension of hidden layer')
    parser.add_argument('--LAYER-NUM', type=int, help='number of LSTM layer')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.set_defaults(
        MODEL = 'LSTM',
        EPOCH = 10,
        BATCH_SIZE = 50,
        TIME_STEP = 10,
        INPUT_SIZE = 8,
        HIDDEN_SIZE = 4,
        LAYER_NUM = 1,
        LR = 0.001
    )
    return parser.parse_args()


class DataFormatting(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label = self.data["id"]
        self.feature = self.data.drop(["id"], axis=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        kind = 1
        if self.label.iloc[i][0] == 'N':
            kind = 0
        d = torch.from_numpy(np.array(self.feature.iloc[i]))
        return d, kind


def main():
    args = args_parse()
    
    trainData = DataFormatting("data/train.csv")
    train_loader = DataLoader(dataset=trainData, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)
    testData = DataFormatting("data/test.csv")
    test_loader = DataLoader(dataset=testData, batch_size=args.BATCH_SIZE , shuffle=True, num_workers=4)

    model = eval(args.MODEL)(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.EPOCH):
        for step, (x, y) in enumerate(train_loader): # x: features of batch data, y: labels of batch data
            b_x = Variable(x.view(-1, args.TIME_STEP, args.INPUT_SIZE))
            b_y = Variable(y)
            output = model(b_x).view(-1, 2)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step % 1000 == 0:
                total_test_loss = []
                suc = 0
                time = 0
                for test_step,(test_x, test_y) in enumerate(test_loader):
                    t_x = Variable(test_x.view(-1, args.TIME_STEP, args.INPUT_SIZE))
                    t_y = Variable(test_y)
                    output_test = model(t_x).view(-1, 2)
                    loss_test = loss_func(output, t_y)
                    total_test_loss.append(loss_test.item())
                    for ind,d in enumerate(test_y):
                        time += 1
                        if np.argmax(output_test[ind].detach().numpy())==test_y[ind]:
                            suc += 1

                accuracy = suc/(time)
                print("epoch:",epoch,"train_loss: {:0.4f} test_loss: {:0.4f} accruacy: {:0.4f}".format(loss.data,np.mean(total_test_loss),accuracy))

if __name__ == "__main__":
    main()