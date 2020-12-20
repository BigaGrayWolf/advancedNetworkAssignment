import numpy as np
import pandas as pd
import argparse, torch
from tqdm import tqdm
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.models import *


def args_parse():
    parser = argparse.ArgumentParser(description='abnormal flow detection')
    parser.add_argument('--MODEL', type=str, help='type of RNN model, option: LSTM, GRU, BiLSTM')
    parser.add_argument('--EPOCH', type=int, help='number of epochs')
    parser.add_argument('--BATCH-SIZE', type=int, help='minibatch size')
    parser.add_argument('--TIME-STEP', type=int, help='number of time steps')
    parser.add_argument('--INPUT-SIZE', type=int, help='dimension of input')
    parser.add_argument('--HIDDEN-SIZE', type=int, help='dimension of hidden layer')
    parser.add_argument('--LAYER-NUM', type=int, help='number of LSTM layer')
    parser.add_argument('--LR', type=float, help='learning rate')
    parser.set_defaults(
        MODEL = 'LSTM',
        EPOCH = 5,
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

def loop_dataset(args, dataloader, model, loss_func, optimizer):
    total_loss = []
    total_acc = []
    all_targets = []
    all_scores = []
    pbar = tqdm(dataloader, unit='batch', ascii=True)
    for (x, y) in pbar:
        b_x = Variable(x.view(-1, args.TIME_STEP, args.INPUT_SIZE))
        b_y = Variable(y)

        output = model(b_x).view(-1, 2)
        all_targets.append(b_y.detach())

        loss = loss_func(output, b_y)
        total_loss.append(loss.item())

        pred = np.argmax(output.detach(), axis=1)
        all_scores.append(pred.detach())

        acc = pred.eq(b_y).sum().item() / float(b_y.size()[0])
        total_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

    avg_loss = np.mean(total_loss)
    avg_acc = np.mean(total_acc)

    all_targets = torch.cat(all_targets).numpy()
    all_scores = torch.cat(all_scores).numpy()
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores)
    auc = metrics.auc(fpr, tpr)

    return [avg_loss, avg_acc, auc]


def main():
    args = args_parse()
    
    trainData = DataFormatting("data/train.csv")
    train_loader = DataLoader(dataset=trainData, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)
    testData = DataFormatting("data/test.csv")
    test_loader = DataLoader(dataset=testData, batch_size=args.BATCH_SIZE , shuffle=True, num_workers=4)

    model = eval(args.MODEL)(args)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    
    for epoch in range(args.EPOCH):
        train_res = loop_dataset(args, train_loader, model, loss_func, optimizer)
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, train_res[0], train_res[1], train_res[2]))
        test_res = loop_dataset(args, test_loader, model, loss_func, optimizer)
        print('\033[92maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_res[0], test_res[1], test_res[2]))
        with open("log/"+"Results_"+args.MODEL+".txt", "a+") as f:
            print('average training of epoch %d: loss %.5f acc %.5f auc %.5f' % (epoch, train_res[0], train_res[1], train_res[2]), file=f)
            print('average test of epoch %d: loss %.5f acc %.5f auc %.5f' % (epoch, test_res[0], test_res[1], test_res[2]), file=f)

if __name__ == "__main__":
    main()