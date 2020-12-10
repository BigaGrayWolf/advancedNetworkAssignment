import os
import dpkt
import math
import numpy as np
import pandas as pd


def parse(filepath, count, dtype, outpath = '../data/misc/bin'):
    ''' preliminarily parse pcap files and extract features 
        including id, time, protocol type, packet length, source and destination port
        input pcap file, output desired DataFrame and number of lines in pcap file
    args:
        filepath: input path of pcap file
        count: id count based on previous pcap files
        dtype: types of data source, 'B' or 'N'
        outpath: output path for binary files
    '''
    f = open(filepath,'rb')
    pcap = dpkt.pcap.Reader(f)
    pts = next(pcap)[0] # previous timestamp

    id_col = [] # id for each line
    ti_col = [] # time interval
    ptc_col = [] # 0 for TCP, 1 for SSH
    len_col = [] # packet length
    sport_col = [] # source port
    dport_col = [] # destination port

    for id, (ts, buf) in enumerate(pcap):
        # id
        id_col.append(dtype+str(id+1+count))
        # time since previous frame in this TCP stream
        ti_col.append(float(format(ts-pts,'.6f')))
        pts = ts
        # protocol, length, sport and dport
        eth = dpkt.ethernet.Ethernet(buf)
        if hasattr(eth.data, 'data'):
            if len(eth.data.data.data) == 0: 
                ptc_col.append(0)
            else:
                ptc_col.append(1)
            tcp = eth.data.data
            sport_col.append(tcp.sport)
            dport_col.append(tcp.dport)
        else: 
            ptc_col.append(0)
            sport_col.append(-1)
            dport_col.append(-1)
        len_col.append(len(eth))
        # binary file
        with open(os.path.join(outpath, dtype+str(id+1+count)+'.bin'), 'wb') as b:
            b.write(eth.pack())

    df = pd.DataFrame()
    df['id'] = id_col
    df['ti'] = ti_col
    df['ptc'] = ptc_col
    df['len'] = len_col
    df['sport'] = sport_col
    df['dport'] = dport_col

    print(filepath + " Done!" )

    return df, id+1+count

def byteEntropy(b):
    dic = {}
    length = 0
    for i in b:
        length +=1
        if i in dic:
            dic[i]+=1
        else:
            dic[i] = 1
    result = 0
    for key in dic:
        p = dic[key]/length
        result += p*math.log2(p)
    return result

def semiByteEntropy(by):
    dic = {}
    length = 0
    before = 0b11110000
    end = 0b00001111
    for i in by:
        a = (i & before)>>4
        b = (i & end)
        length +=1
        if a in dic:
            dic[a]+=1
        else:
            dic[a]=1
        if b in dic:
            dic[b]+=1
        else:
            dic[b]=1
    length *=2
    result = 0
    for key in dic:
        p = dic[key] / length
        result += p * math.log2(p)
    return result

def bitEntropy(b):
    dic = {}
    length = 0
    pattern = 0b1
    for i in b:
        for j in range(8):
            temp = (i>>j)&pattern
            if temp in dic:
                dic[temp]+=1
            else:
                dic[temp]=1
            length+=1
    result = 0
    for key in dic:
        p = dic[key] / length
        result += p * math.log2(p)
    return result

def calcEnt(filedir="../data/misc"):
    '''calculte byteEntropy, semiByteEntropy, bitEntropy features'''
    d = {"id":[],"byteEntropy":[],"semiByteEntropy":[],"bitEntropy":[]}
    for filename in os.listdir(filedir):
        with open(os.path.join(filedir,filename),"rb") as f:
            b = f.read()
            d["id"].append(filename.split(".")[0])
            d["byteEntropy"].append(byteEntropy(b))
            d["semiByteEntropy"].append(semiByteEntropy(b))
            d["bitEntropy"].append(bitEntropy(b))
    dataFrame = pd.DataFrame(d)
    dataFrame.to_csv(os.path.join(filedir,"entropy.csv"),index=False)

def dealCSV(filename1="normal.csv",filename2="brute.csv",filename3="entropy.csv",filedir="../data/misc"):
    '''concat file1 and file2 then merge file3'''
    n = pd.read_csv(os.path.join(filedir,filename1))
    b = pd.read_csv(os.path.join(filedir,filename2))
    c = pd.concat([n,b],ignore_index=True)
    e = pd.read_csv(os.path.join(filedir,filename3))
    r = pd.merge(c,e,on="id")
    r.to_csv(os.path.join(filedir,"generalTable.csv"),index=False)


def main():
    if not os.path.exists("./data"):
        os.mkdir("./data")
    if not os.path.exists("./data/misc"):
        os.mkdir("./data/misc")
    if not os.path.exists("./data/misc/bin"):
        os.mkdir("./data/misc/bin")
    
    # preliminarily parse all the pcap files
    df1, count1 = parse('../data/raw/brute/pkts_20201011-115843.pcap', 0, 'B')
    df2, count2 = parse('../data/raw/brute/pkts_20201011-120011.pcap', count1, 'B')
    df3, count3 = parse('../data/raw/brute/pkts_20201011-164816.pcap', count2, 'B')
    df4, count4 = parse('../data/raw/brute/pkts_20201011-173645.pcap', count3, 'B')
    df5, count5 = parse('../data/raw/brute/pkts_20201012-005541.pcap', count4, 'B')
    df6, count6 = parse('../data/raw/brute/pkts_20201012-012714.pcap', count5, 'B')
    df7, count7 = parse('../data/raw/brute/pkts_20201012-045617.pcap', count6, 'B')
    df_b = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
    df_b.to_csv('./data/misc/brute.csv', sep=',', index=False, header=True)

    df8, count8 = parse('../data/raw/normal/pkts_20201011-115531.pcap', 0, 'N')
    df9, count9 = parse('../data/raw/normal/pkts_20201011-172425.pcap', count8, 'N')
    df_n = pd.concat([df8, df9], ignore_index=True)
    df_n.to_csv('./data/misc/normal.csv', sep=',', index=False, header=True)

    print("number of processed packets in each brute pcap file:\n\
    num1=%d, num2=%d, num3=%d, num4=%d, num5=%d, num6=%d, num7=%d"%
    (count1, count2-count1, count3-count2, count4-count3, count5-count4, count6-count5, count7-count6))
    print("number of processed packets in each normal pcap file:\n\
    num8=%d, num9=%d"%(count8, count9-count8))

    # add entropy features
    calcEnt()
    dealCSV()

    # normalize each colunmns in generalTable.csv
    data = pd.read_csv("../data/misc/generalTable.csv")
    for name in data.columns:
        if name=="id":
            continue
        data[name] = (data[name]-data[name].min())/(data[name].max()-data[name].min())
    data.to_csv("../data/misc/normalize.csv",index=False)
    
    # split data into train.csv and test.csv
    data = pd.read_csv("../data/misc/normalize.csv")
    print(data.iloc[99065:99069])
    n_train = data.iloc[:50000]
    n_test = data.iloc[50000:51000]
    b_train = data.iloc[99066:149066]
    b_test = data.iloc[149066:150066]
    train = pd.concat([n_train,b_train])
    test = pd.concat([n_test,b_test])
    train.to_csv("../data/train.csv",index=False)
    test.to_csv("../data/test.csv",index=False)

if __name__ == "__main__":
    main()