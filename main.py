import pcap
import threading
import dpkt
import os
import time
import paramiko
import random
import sys

class normalLogin():
    def __init__(self,user_name,password,targetIP):
        self.user_name = user_name
        self.password = password
        self.targetIP = targetIP

    def loginOP(self):
        fake_password = [self.password+"{}".format(i) for i in range(11)]
        count = random.randint(1,10)
        l = ["mkdir wang", "cd wang", "touch abc", "touch kkk", "touch kkja", "touch tidj", "touch smb",
             "mkdir shao", "cd shao", "touch abc", "touch kkk", "touch kkja", "touch tidj", "touch smb",
             "mkdir feng", "cd feng", "touch abc", "touch kkk", "touch kkja", "touch tidj", "touch smb",
             "cd..", "cd..", "cd..", "rm -rf wang"]
        for i in range(count):
            ssh = paramiko.SSHClient()
            try:

                if i==count-1:
                    ssh.connect(self.targetIP,22,self.user_name,self.password,timeout=5)
                    for cmd in l:
                        ssh.exec_command(cmd)
                        time.sleep(random.random())
                    ssh.close()
                else:
                    ssh.connect(self.targetIP, 22, self.user_name, fake_password[i] , timeout=5)
                    ssh.close()
            except:
                ssh.close()




class bruteAttack():
    def __init__(self,isSuccess,user_name, targetIP):
        self.isSuccess = isSuccess
        self.user_name = user_name
        self.targetIP = targetIP

    def attackString(self):
        if self.isSuccess:
            s = "hydra -l "+ self.user_name +" -P password_suc.txt -t 4 -I ssh://" + self.targetIP
        else:
            s = "hydra -l " + self.user_name + " -P password_fail.txt -t 4 -I ssh://" + self.targetIP
        return s

    #the operation after login
    def afterLogin(self):
        l = ["mkdir wang","cd wang","touch abc","touch kkk","touch kkja","touch tidj","touch smb",
             "mkdir shao","cd shao","touch abc","touch kkk","touch kkja","touch tidj","touch smb",
             "mkdir feng","cd feng","touch abc","touch kkk","touch kkja","touch tidj","touch smb",
             "cd..","cd..","cd..","rm -rf wang"]
        return "&&".join(l)

    def run(self):
        if os.path.isfile("./hydra.restore"):
            os.remove("./hydra.restore")

        if self.isSuccess:
            os.popen(self.attackString()+"&&"+self.afterLogin()+"&&exit")
        else:
            os.popen(self.attackString())


#pkt is a sniffer
#save_dir is the dir of the pcap file
class sniffer(threading.Thread):
    def __init__(self, pkt ,save_dir="./"):
        threading.Thread.__init__(self)
        self.save_dir = save_dir
        self.pkt = pkt

    def save(self,pcap_file,writer,filepath):
        try:
            for ptime,pdata in self.pkt:
                writer.writepkt(pdata,ptime)
        except KeyboardInterrupt as e:
            self.pkt.close()
        finally:
            writer.close()
            pcap_file.close()

    def run(self):
        filepath = os.path.join(self.save_dir,"pkts_{}.pcap".format(time.strftime("%Y%m%d-%H%M%S",time.localtime())))
        pcap_file = open(filepath,"ab+")
        writer = dpkt.pcap.Writer(pcap_file)
        self.save(pcap_file,writer,filepath)


def createRandomString(n):
    char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    s = ''
    for i in range(n):
        s = s+random.choice(char)
    return s

def createPasswordFile(password):
    if not os.path.isfile("password_suc.txt"):
        pwd_list = []
        for i in range(100):
            pwd_list.append(createRandomString(8))

        with open("password_suc.txt","w") as f:
            pwd_list.append(password)
            f.write("\n".join(pwd_list))
        with open("password_fail.txt","w") as f:
            f.write("\n".join(pwd_list))

def deletePasswordFile():
    if os.path.isfile("password_suc.txt"):
        os.remove("password_suc.txt")
        os.remove("password_fail.txt")

# Press the green button in the gutter to run the script.
def run(count,pkt,username,password,kind,save_path,targetIP,isSuccess):

    ear = sniffer(pkt, save_path)
    ear.start()
    for i in range(count):
        if kind == "normal":
            normalLogin(username, password, targetIP).loginOP()
        else:
            if isSuccess == "success":
                bruteAttack(True, username, targetIP).run()
            else:
                bruteAttack(False, username, targetIP).run()


if __name__ == '__main__':
    username = sys.argv[1]
    password = sys.argv[2]
    targetIP = sys.argv[3]
    kind = sys.argv[4]
    count = int(sys.argv[5])

    createPasswordFile(password)
    dev = pcap.findalldevs()[0]
    pkt = pcap.pcap(dev, promisc=True, immediate=True, timeout_ms=5)
    pkt.setfilter('tcp port 22')
    if kind=="normal":
        run(count,pkt, username, password, "normal", "./normal/", targetIP, "success")
    else:
        p = random.random()
        if p>0.1:
            run(count,pkt, username, password, "brute", "./brute/", targetIP, "fail")
        else:
            run(count,pkt, username, password, "brute", "./brute/", targetIP, "success")


