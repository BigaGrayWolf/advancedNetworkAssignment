import pandas as pd


# data = pd.read_csv("./generalTable.csv")
#
# # 对data中所有列做归一化
# for name in data.columns:
#     if name=="id":
#         continue
#     data[name] = (data[name]-data[name].min())/(data[name].max()-data[name].min())
#
# data.to_csv("./normalize.csv",index=False)


# 把data分成两个文件 train.csv test.csv


data = pd.read_csv("./normalize.csv")
import numpy as np
print(data.iloc[99065:99069])

n_train = data.iloc[:50000]
n_test = data.iloc[50000:51000]
b_train = data.iloc[99066:149066]
b_test = data.iloc[149066:150066]
train = pd.concat([n_train,b_train])
test = pd.concat([n_test,b_test])
train.to_csv("./train.csv",index=False)
test.to_csv("./test.csv",index=False)
