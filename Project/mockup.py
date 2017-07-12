import random
import numpy as np
import os
import csv
from pprint import pprint
import re
from itertools import zip_longest
from entity import Entity
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
from sklearn import datasets, linear_model
from copy import deepcopy

def cleanData():
	def isIP(iocs):
		ips=[]

		for ioc in iocs[1:]:
			if re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',ioc) != None:
				ips.append(ioc)
			elif re.match('([a-fA-F\d]{32})',ioc) != None:
				ips.append(None)
			else:
				ips.append(None)
		ips.insert(0,'IP Classification')
		return ips

	def isURL(iocs):
		urls=[]

		for ioc in iocs[1:]:
			if re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',ioc) != None:
				urls.append('')
			elif re.match('([a-fA-F\d]{32})',ioc) != None:
				urls.append(None)
			else:
				if ioc.find('.') != -1:
			 		urls.append(ioc)
				else:
	 				urls.append(None)
		urls.insert(0,'URL Classification')

		return urls

	def isHash(iocs):
		hashs=[]

		for ioc in iocs[1:]:
			if re.match('([a-fA-F\d]{32})',ioc) != None:
				hashs.append(ioc)
			else:
				hashs.append(None)
		hashs.insert(0,'Hash Classification')
		return hashs

	def knownThreat(titles):
		threats = []
		flag = False
		commonThreats = ['WannaCry', 'RAT','APT']
		for title in titles[1:]:
			if re.match('([a-fA-F\d]{32})',title) != None:
				threats.append("Hash")
			else:
				for threat in commonThreats:
					if title.find(threat) != -1:
						threats.append(threat)
						flag=True
						break
				if flag == False:
					threats.append(None)
				flag = False
		threats.insert(0,'Known Threat')
		return threats

	def classifDate(dates):
		date_split = []
		#for title
		for date in dates[1:]:
			spl_date = date.split()
			#[:-1] is to remove the comma at then end of the parse
			if int(spl_date[2][:-1]) < 2017:
				date_split.append(True)
			else:
				date_split.append(None)
		date_split.insert(0,'Current Threat')
		return date_split

	def threatLevel(levels):
		low = []
		med = []
		high = []
		undef = []

		for level in levels[1:]:
			if level == 'Low':
				low.append('Low')
				med.append(None)
				high.append(None)
				undef.append(None)

			elif level == 'Medium':
				low.append(None)
				med.append('Medium')
				high.append(None)
				undef.append(None)

			elif level == 'High':
				low.append(None)
				med.append(None)
				high.append('High')
				undef.append(None)

			elif level == 'Undefined':
				low.append(None)
				med.append(None)
				high.append(None)
				undef.append('Undefined')
			else:
				low.append(None)
				med.append(None)
				high.append(None)
				undef.append(None)

		low.insert(0,'If Low')
		med.insert(0,'If Medium')
		high.insert(0,'If High')
		undef.insert(0,'If Undefined')
		return low,med,high,undef



	file = open('ioc_hunt_long.csv','r')

	dates=[]
	iocs=[]
	titles=[]
	levels=[]
	classifs=[]

	reader = csv.reader(file, delimiter=',')

	for row in reader:
	    dates.append(row[0])
	    iocs.append(row[1])
	    titles.append(row[2])
	    levels.append(row[3])
	    classifs.append(row[4])


	low, med, high, undef = threatLevel(levels)
	clean_data = zip_longest(dates, classifDate(dates), isIP(iocs), isURL(iocs), isHash(iocs), knownThreat(titles), levels, low, med, high, undef)
	outfile = open("clean_data.csv", "w",newline='')
	writer = csv.writer(outfile)
	writer.writerows(clean_data)
	outfile.close()

	f = open('clean_data.csv', 'r')
	reader = csv.reader(f, delimiter=',')

	entList = []
	names = f.readline()

	for row in reader:
	    entList.append(Entity(row))

	return entList


	

def initData():
	entList = cleanData()
	return entList

def enrich(ent,action,randomEnrichemnts):
	if action == 0:
		if ent.state[ent.state.size-3] != 1:
			ent.state[ent.state.size-3] = randomEnrichemnts[0]
			ent.updatePolicy(action)
	elif action == 1:
		if ent.state[ent.state.size-2] != 1:
			ent.state[ent.state.size-2] = randomEnrichemnts[1]
			ent.updatePolicy(action)
	elif action == 2:
		if ent.state[ent.state.size-1] != 1:
			ent.state[ent.state.size-1] = randomEnrichemnts[2]
			ent.updatePolicy(action)
	elif action == 3:
		if ent.state[ent.state.size-6] != 1:
			ent.state[ent.state.size-6] = randomEnrichemnts[3]
			ent.updatePolicy(action)
	elif action == 4:
		if ent.state[ent.state.size-5] != 1:
			ent.state[ent.state.size-5] = randomEnrichemnts[4]
			ent.updatePolicy(action)
	elif action == 5:
		if ent.state[ent.state.size-4] != 1:
			ent.state[ent.state.size-4] = randomEnrichemnts[5]
			ent.updatePolicy(action)

	return ent.state

def getReward(state,oldState):

	if np.count_nonzero(state[-6:]) > np.count_nonzero(oldState[-6:]):

		if np.count_nonzero(state[-6:]) == 0:
			return 0
		elif np.count_nonzero(state[-6:]) == 1:
			return (1)
		elif np.count_nonzero(state[-6:]) == 2:
			return (2)
		elif np.count_nonzero(state[-6:]) == 3:
			return (3)
		elif np.count_nonzero(state[-6:]) == 4:
			return (4)
		elif np.count_nonzero(state[-6:]) == 5:
			return (5)
		elif np.count_nonzero(state[-6:]) == 6:
			return (6)
	return -10

def assocRandomHelper(lst,ele):
	for i in lst:
		if np.array_equal(i[0] , ele): 
			return i[1]
	return None	

def assocRandom(ents):
	randAssoc= []
	for ent in ents:
		temp = assocRandomHelper(randAssoc,ent.state[:11])
		if temp == None:
			randAssoc.append((ent.state[:11],randoEnrich()))
		else:
			randAssoc.append((ent.state[:11],temp))
	return randAssoc


def randoEnrich():
	randomEnrichemnts = []
	for i in range(6):
		randomEnrichemnts.append(np.random.randint(0,2))
	return randomEnrichemnts




def virusTotal():
	score =np.random.randint(0,2)
	# print ('Hit VT. Score is: ',score)
	return score	

def whiteList():
	score =np.random.randint(0,2)
	# print ('Hit Whitelist. Score is: ',score)
	return score

def passiveTotal():
	score =np.random.randint(0,2)
	# print ('Hit PT. Score is: ',score)
	return score

def a():
	score =np.random.randint(0,2)
	# print ('Hit A. Score is: ',score)
	return score

def b():
	score =np.random.randint(0,2)
	# print ('Hit B. Score is: ',score)
	return score

def c():
	score =np.random.randint(0,2)
	# print ('Hit C. Score is: ',score)
	return score


def dispData(state):
	print (state)

def initNN(inputNodes,outputNodes):
	model = Sequential()
	model.add(Dense(164, init='lecun_uniform', input_shape=(inputNodes,)))
	model.add(Activation('relu'))

	model.add(Dense(150, init='lecun_uniform'))
	model.add(Activation('relu'))

	model.add(Dense(outputNodes, init='lecun_uniform'))
	model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)
	return model

numRuns = 0
inputNodes = 17
outputNodes = 6
model = initNN(inputNodes,outputNodes)
ents = initData()

enrichMap=(assocRandom(ents))

x= 0
rewards = [] #keep track of rewards to analyze
epochs = 329
gamma = 1 #since it may take several moves to goal, making gamma high
learning_rate = 1
epsilon = 1

for i in range(epochs):
   
    print('\nGame',i)
    totalReward = 0
    randActions = 0 #keep track of number of random actions
    prevAct = 0
    numRuns=0
    toolChecked =[0,0,0,0,0,0]
    ent = ents[i]
    print ('State:',ent.state[:11])    
    # dispData(ent.state)  
    status = 1
    #while game still in progress
    randomEnrichemnts =enrichMap[i][1] 
    while(status == 1):
    	
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(ent.state.reshape(1,inputNodes), batch_size=1)

        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,outputNodes)
            randActions += 1
        else: #choose best action from Q(s,a) values
        	action = (np.argmax(qval))
        # print('Policy Checked:',toolChecked,'Enrichments:',randomEnrichemnts)	
		       
        oldState = deepcopy(ent.state)
       	if toolChecked[action] != 1:
	        new_state = enrich(ent, action , randomEnrichemnts)
	        reward = getReward(new_state,oldState)
	        totalReward += reward
	        toolChecked[action]=1
	        # print ('new',new_state,'\nold',oldState)
	        # print('\nAction:',action,'\tReward:',reward,'\tPrint Q table', qval)
	        # dispData(new_state)

	        #Get max_Q(S',a)
	        newQ = model.predict(new_state.reshape(1,inputNodes), batch_size=1)
	        maxQ = np.max(newQ)

	        y = np.zeros((1,outputNodes))
	        y[:] = qval[:]
	        
	        if reward != 1:  #non-terminal state
	            update = learning_rate * (reward + (gamma * maxQ))
	        
	        else: #terminal state
	            update = reward

	        y[0][action] = update #target output
	        model.fit(ent.state.reshape(1,inputNodes), y, batch_size=1, nb_epoch=1,verbose = False)
	        ent.state = new_state
        
        numRuns+=1
        if reward >= 3 or numRuns > 20:		
            status = 0
            print('Policy:',ent.policy)
            print('Rand Actions:', randActions)
            print('Print Q table', qval)
            print('Random Enrichment Decisions', randomEnrichemnts)
            if np.array_equal(ent.state[:11] ,  [ 1, 0,  1,  0,  0,  1,  1,  0,  0,  1,  0]):
            	z = randomEnrichemnts
            	print('HIT\n\n\n')
        # clear_output(wait=True)
    		
    if epsilon > 0.1:
        epsilon -= (1/epochs)




q1 = model.predict(np.array([1, 0,  1,  0,  0,  1,  1,  0,  0,  1,  0, z[0], z[1], z[2], z[3], z[4], z[5]]).reshape(1,inputNodes), batch_size=1)

q2 = model.predict(np.array([1,  0,  1,  0,  0,  1,  1,  1,  0,  0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,inputNodes), batch_size=1)

q3 = model.predict(np.array([1,  0,  1,  0,  0,  1,  1,  1,  0,  0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,inputNodes), batch_size=1)

print('Q1',q1)
print('Q2',q2)
print('Q3',q2)