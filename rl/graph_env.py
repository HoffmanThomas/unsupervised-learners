
import numpy as np

#number of states 
n = 3

#list for storing the states 
states = []

#state space 
state_space = np.zeros(shape = (n,n))

#init the states and the state space
for i in range(0,n):

	#create the list of states
	states.append(str("p"+ str(i)))

	for j in range(0,n):
		
		#probability of going from state i to state j 
		state_space[i][j] = 1/n

	
print('\nProcesses: \n', states, '\n')
print('State Space: \n',state_space)

for i in range(0, state_space.shape[1]):
	print(i)
	#state_space.sum(axis = i)