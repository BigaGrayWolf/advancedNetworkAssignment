import dpkt
import os
import pandas as pd

def parse(filepath, count, dtype, outpath = './data/bin'):
    '''input pcap file, output desired DataFrame and number of lines in pcap file
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

def main():
    if not os.path.exists("./data"):
        os.mkdir("./data")
    if not os.path.exists("./data/bin"):
        os.mkdir("./data/bin")
    
    df1, count1 = parse('./brute/pkts_20201011-115843.pcap', 0, 'B')
    df2, count2 = parse('./brute/pkts_20201011-120011.pcap', count1, 'B')
    df3, count3 = parse('./brute/pkts_20201011-164816.pcap', count2, 'B')
    df4, count4 = parse('./brute/pkts_20201011-173645.pcap', count3, 'B')
    df5, count5 = parse('./brute/pkts_20201012-005541.pcap', count4, 'B')
    df6, count6 = parse('./brute/pkts_20201012-012714.pcap', count5, 'B')
    df7, count7 = parse('./brute/pkts_20201012-045617.pcap', count6, 'B')
    df_b = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
    df_b.to_csv('./data/brute.csv', sep=',', index=False, header=True)

    df8, count8 = parse('./normal/pkts_20201011-115531.pcap', 0, 'N')
    df9, count9 = parse('./normal/pkts_20201011-172425.pcap', count8, 'N')
    df_n = pd.concat([df8, df9], ignore_index=True)
    df_n.to_csv('./data/normal.csv', sep=',', index=False, header=True)

    print("number of processed packets in each brute pcap file:\n\
    num1=%d, num2=%d, num3=%d, num4=%d, num5=%d, num6=%d, num7=%d"%
    (count1, count2-count1, count3-count2, count4-count3, count5-count4, count6-count5, count7-count6))
    print("number of processed packets in each normal pcap file:\n\
    num8=%d, num9=%d"%(count8, count9-count8))

if __name__ == "__main__":
    main()