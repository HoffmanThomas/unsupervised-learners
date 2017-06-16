import subprocess
import sys

command = 'Rscript'
path2script = 'C:/Users/thoffma/Desktop/unsupervised-learners/py_r/testScript.R'

# Variable number of args in a list
args = ['11', '3', '9', '42']

# Build subprocess command
cmd = [command, path2script] + args
print(cmd)

# check_output will run the command and store to result
x = subprocess.check_output(cmd, shell=True)

print('The maximum of the numbers is:', x)