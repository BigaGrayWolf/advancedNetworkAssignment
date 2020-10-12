import sys
import random
import os

if __name__ == "__main__":
    user_name = sys.argv[1]
    password = sys.argv[2]
    ip = sys.argv[3]#192.168.139.128
    normalTime = sys.argv[4]
    bruteTime = sys.argv[5]

    if not os.path.exists("./normal/"):
        os.mkdir("./normal/")

    if not os.path.exists("./brute/"):
        os.mkdir("./brute/")

    for i in range(int(normalTime)):
        os.popen("python main.py {} {} normal ./normal/ {} success".format(user_name,password,ip))

    for i in range(int(bruteTime)):
        p = random.random()
        if p>0.1:
            os.popen("python main.py {} {} brute ./brute/ {} fail".format(user_name,password,ip))
        else:
            os.popen("python main.py {} {} brute ./brute/ {} success".format(user_name, password,ip))
