from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import numpy as np
import torch
from torch.autograd import Variable
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


trainData = myDataSet("./train.csv")
train_loader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testData = myDataSet("./test.csv")
test_loader = DataLoader(dataset=testData, batch_size=BATCH_SIZE , shuffle=True, num_workers=0)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYER_NUM,
            batch_first=True,  # 把batch放在第一个维度
        )
        self.ou = nn.Linear(HIDDEN_SIZE, 2)
    def forward(self, x):
        x = x.float()
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
        out = self.ou(r_out)
        return out


rnn = RNN()
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


## 输出
# RNN(
#   (rnn): LSTM(8, 4, batch_first=True)
#   (ou): Linear(in_features=4, out_features=2, bias=True)
# )
# epoch: 0 train_loss: 0.7628 test_loss: 0.7249 accruacy: 0.5000
# epoch: 0 train_loss: 0.5771 test_loss: 0.7412 accruacy: 0.7345
# epoch: 1 train_loss: 0.3786 test_loss: 0.8610 accruacy: 0.8860
# epoch: 1 train_loss: 0.1654 test_loss: 1.1039 accruacy: 0.9760
# epoch: 2 train_loss: 0.0876 test_loss: 1.4371 accruacy: 0.9860
# epoch: 2 train_loss: 0.0683 test_loss: 1.8386 accruacy: 0.9995
# epoch: 3 train_loss: 0.0348 test_loss: 2.0879 accruacy: 1.0000
# epoch: 3 train_loss: 0.0195 test_loss: 2.3375 accruacy: 1.0000
# epoch: 4 train_loss: 0.1141 test_loss: 2.5971 accruacy: 1.0000
# epoch: 4 train_loss: 0.0463 test_loss: 2.9293 accruacy: 1.0000
# epoch: 5 train_loss: 0.0107 test_loss: 3.0831 accruacy: 1.0000
# epoch: 5 train_loss: 0.1861 test_loss: 3.1955 accruacy: 1.0000
# epoch: 6 train_loss: 0.0034 test_loss: 3.4216 accruacy: 1.0000
# epoch: 6 train_loss: 0.0063 test_loss: 3.5422 accruacy: 1.0000
# epoch: 7 train_loss: 0.0031 test_loss: 3.8671 accruacy: 1.0000
# epoch: 7 train_loss: 0.1633 test_loss: 3.8002 accruacy: 1.0000
# epoch: 8 train_loss: 0.0048 test_loss: 3.7631 accruacy: 1.0000
# epoch: 8 train_loss: 0.0021 test_loss: 3.7665 accruacy: 1.0000
# epoch: 9 train_loss: 0.0069 test_loss: 3.7164 accruacy: 1.0000
# epoch: 9 train_loss: 0.0021 test_loss: 4.1853 accruacy: 1.0000