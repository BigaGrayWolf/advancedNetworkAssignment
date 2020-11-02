import pandas as pd
import os
import math

filedir = "G:/data/bin"
filedir2 = "G:/data"
def dealCSV(filename1="normal.csv",filename2="brute.csv",filename3="entropy.csv"):
    n = pd.read_csv(os.path.join(filedir2,filename1))
    b = pd.read_csv(os.path.join(filedir2,filename2))
    c = pd.concat([n,b],ignore_index=True)
    e = pd.read_csv(os.path.join(filedir2,filename3))
    r = pd.merge(c,e,on="id")
    r.to_csv(os.path.join(filedir2,"generalTable.csv"),index=False)
dealCSV()

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

def list_files(filedir):
    d = {"id":[],"byteEntropy":[],"semiByteEntropy":[],"bitEntropy":[]}
    for filename in os.listdir(filedir):
        with open(os.path.join(filedir,filename),"rb") as f:
            b = f.read()

            d["id"].append(filename.split(".")[0])
            d["byteEntropy"].append(byteEntropy(b))
            d["semiByteEntropy"].append(semiByteEntropy(b))
            d["bitEntropy"].append(bitEntropy(b))


    dataFrame = pd.DataFrame(d)
    dataFrame.to_csv("G:/data/entropy.csv",index=False)



#list_files(filedir)
