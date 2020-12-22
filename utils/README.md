#

**How to use data_generator.py ?**

Enter the directory where data_generator.py is located,
input the order in your interminal:

> python data_generator.py *username* *password* *targetIP* *kind* *loginTime*

NOTES:

*username*, *password* and *targetIP* is for your ssh login.
You may build the ssh-server in the other virtual machine.
*kind* is either "normal" or "brute". "normal" stands for the
normal user mode, and "brute" stands for the brute attack mode.

Here are two examples:
> python data_generator.py WSF WSF 192.168.1.1 normal 10 \
> python data_generator.py WSF WSF 192.168.1.1 brute 40

The program cannot end itself. You must use "ctrl+c" to end the program
in this version.

**How to use data_preprocessor.py ?**

Enter the directory where data_preprocessor.py is located,
input the order in your interminal:

> python data_generator.py

You will find train.csv and test.csv generated under the "data" directory.
