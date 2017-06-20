import subprocess
import sys

command = 'Rscript'
path2script = 'C:/Users/thoffma/Desktop/unsupervised-learners/py_r/BeerNN_in_R.R'

# Variable number of args in a list
args = []

# Build subprocess command
cmd = [command, path2script] + args
print(cmd)

# check_output will run the command and store to result
x = subprocess.check_output(cmd, shell=True)

print('yhat:', x)