
# 1.基本说明

本实验程序实现了基于RNN模型的SSH协议暴力破解流量分析。

* 首先搭建SSH服务器，编写脚本（utils/data_generator.py）模拟生成正常流量和暴力破解流量（data/raw）。
* 接着以数据包为单位提取间隔时间、协议等相应特征，生成训练/测试数据（utils/data_preprocessor.py）。
* 最后利用GRU、LSTM、BiLSTM三种RNN模型（model/models.py），
  对训练/测试数据进行正常流量和暴力破解流量二分类预测（train_main.py）,
  并输出记录训练/测试结果（log）。

# 2.运行环境说明

本实验程序采用Python语言编写，主要使用如下软件包：

* pcap
* dpkt
* paramiko
* tqdm
* numpy
* pandas
* sklearn
* pytorch
* torchvision

# 3.启动命令

数据生成和预处理文件的启动运行请参考utils目录下的README.md

RNN模型训练测试文件的启动命令为：

> python train_main.py --MODEL XXX

其中'XXX'可以为'GRU'、'LSTM'或'BiLSTM'

# 4.代码目录结构

|——  README.md

|——  train_main.py

|——  data

    |——  raw

    |——  train.csv

    |——  test.csv

|——  model

    |——  models.py

|——  utils

    |——  README.md

    |——  data_generator.py

    |——  data_preprocessor.py

|——  log
