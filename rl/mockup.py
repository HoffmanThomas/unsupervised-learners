import random
import numpy as np
import os 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output


def init_data():
	state = np.array([1,1,1,0,0,0])
	return state

def enrich(state,action):
	if action == 0:
		state[3] = 1
	elif action == 1:
		state[4] = 1
	elif action == 2:
		state[5] = 1

	return state

def getReward(state):
	classify= np.random.randint(1,100)

	if classify <= 50:
		return -10
	if classify >=90:
		return 10

	return -1 

def virusTotal(state):
	if random.random() > .8:
		score = np.random.randint(0,66)
	else:
		score = np.random.randint(0,5)

	percentage = score/65
	return percentage

def whiteList(state):
	if random.random() > .8:
		return True
	else:
		return False

def whoIs(state):
	if random.random() > .8:
		return True
	else:
		return False


def dispData(state):
	print (state)

def init_nn():
	model = Sequential()
	model.add(Dense(164, init='lecun_uniform', input_shape=(6,)))
	model.add(Activation('relu'))

	model.add(Dense(150, init='lecun_uniform'))
	model.add(Activation('relu'))

	model.add(Dense(3, init='lecun_uniform'))
	model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)
	return model



model = init_nn()

epochs = 1000
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
for i in range(epochs):
    
    state = init_data()
    print('\n\n\nStart Game',i)
    dispData(state)    
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,6), batch_size=1)

        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,3)
        else: #choose best action from Q(s,a) values
        	print('Chosing from Qtable')
        	action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = enrich(state, action)
        #Observe reward
        reward = getReward(new_state)
        print('\nAction:',action,'\tReward:',reward,'\tPrint Q table', qval)
        dispData(new_state)
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,6), batch_size=1)
        maxQ = np.max(newQ)

        y = np.zeros((1,3))
        y[:] = qval[:]
        if reward == -1: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        # print("Game #: %s" % (i,))
   

        model.fit(state.reshape(1,6), y, batch_size=1, nb_epoch=1,verbose = False)
        state = new_state
        if reward != -1:
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1:
        epsilon -= (1/epochs)
