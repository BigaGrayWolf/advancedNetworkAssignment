how to use main.py to finish the attack?
input the order in your interminal,there are two cases:

for the fisrt case,you want to login as a normal user:
python main.py username password targetIP kind loginTime

NOTES:
the username ,the password and the targetIP is for your ssh login.
you may build the ssh-server in the other virtual machine.
the kind is either "normal" or "brute".the "normal" stands for the 
normal user mode,and the "brute" stands for the brute attack mode.

here are two examples
python main.py WSF WSF 192.168.1.1 normal 10
python main.py WSF WSF 192.168.1.1 brute 40

the program cannot end itself.you must use "ctrl+c" to end the program
in this version.

