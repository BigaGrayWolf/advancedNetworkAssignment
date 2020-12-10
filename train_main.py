import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.models import *

#Hyper Parameters
EPOCH = 10
BATCH_SIZE = 50
TIME_STEP = 10  # 总共可以看到多少个时间节点
INPUT_SIZE = 8  # 每个时间节点的维度
HIDDEN_SIZE = 4  # lstm中隐藏层的大小 64<16
LAYER_NUM = 1  # lstm中层数
LR = 0.001

class myDataSet(Dataset):
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
    trainData = myDataSet("./train.csv")
    train_loader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testData = myDataSet("./test.csv")
    test_loader = DataLoader(dataset=testData, batch_size=BATCH_SIZE , shuffle=True, num_workers=0)

    model = LSTM()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, TIME_STEP, INPUT_SIZE))
            b_y = Variable(y)
            output = rnn(b_x).view(-1, 2)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if step % 1000 == 0:
                total_test_loss = []
                suc = 0
                time = 0
                for test_step,(test_x, test_y) in enumerate(test_loader):
                    t_x = Variable(test_x.view(-1, TIME_STEP, INPUT_SIZE))
                    t_y = Variable(test_y)
                    output_test = rnn(t_x).view(-1, 2)
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