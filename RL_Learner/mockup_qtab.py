import random
import numpy as np
import csv
from pprint import pprint
import re
from itertools import zip_longest
from entity import Entity
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import RMSprop
# from IPython.display import clear_output
from sklearn import datasets, linear_model
from copy import deepcopy
from tkinter import *
from PIL import ImageTk, Image
import os, time



def cleanData(alerts):
	"""This function will parse a csv of alerts into columns and then expand 
	   some columns to one hot encode them

	Args: 
		alerts: string name of the csv of alerts
	Returns:
		A list of entity objects containing the broken down CSV info 
	"""
	
	def isIP(iocs):
		"""Will look through a column and put all of the IPs in that column into an array
		   
		Args: 
			iocs: Column containing a combo of hashes, URLs, and IPs

		Returns:
			A list containing just the IPs in their original location with None in the place of everything else
		"""
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
		"""Will look through a column and put all of the URLs in that column into an array
		   
		Args: 
			iocs: Column containing a combo of hashes, URLs, and IPs

		Returns:
			A list containing just the URLs in their original location with None in the place of everything else
		"""
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
		"""Will look through a column and put all of the hashes in that column into an array
		   
		Args: 
			iocs: Column containing a combo of hashes, domains, and IPs

		Returns:
			A list containing just the hashes in their original location with None in the place of everything else
		"""
		hashs=[]

		for ioc in iocs[1:]:
			if re.match('([a-fA-F\d]{32})',ioc) != None:
				hashs.append(ioc)
			else:
				hashs.append(None)
		hashs.insert(0,'Hash Classification')
		return hashs

	def knownThreat(titles,commonThreats):
		"""Will look through a column and will identify any keyword in commonThreats 
		   
		Args: 
			titles: Column containing all the titles of alerts
			commonThreats: List of keywords to classify known threats 

		Returns:
			A list containing just the known threats in their original location with None in the place of everything else
		"""
		threats = []
		flag = False
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

	def classifDate(dates,decisionDate):
		"""Will look through a column of dates classify before and after a certain year
		   
		Args: 
			dates: Column containing all of the dates
			decisionDate: date to chose before and after to make a classification 
		Returns:
			A list containing a 1 for after the decisionDate and 0 for after
		"""
		date_split = []
		#for title
		for date in dates[1:]:
			spl_date = date.split()
			#[:-1] is to remove the comma at then end of the parse
			if int(spl_date[2][:-1]) < decisionDate:
				date_split.append(True)
			else:
				date_split.append(None)
		date_split.insert(0,'Current Threat')
		return date_split

	def threatLevel(levels):
		"""Will look through a column of misp threat leveles and one hot encode into 4 columns with just 
		   low,med,high and undefined in their own column
		   
		Args: 
			levels: Column containing all of the misp threat levels
		Returns:
			4 lists containing low, med ,high, and undefined in different columns in their original location 
			with None in the place of everything else
		"""
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



	file = open(alerts,'r')

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
	clean_data = zip_longest(dates, classifDate(dates,2017), isIP(iocs), isURL(iocs), isHash(iocs), knownThreat(titles,['WannaCry', 'RAT','APT']), levels, low, med, high, undef)
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------
	

def initData():
	"""calls cleanData() to initialize the entList 
		   
		Args: 
			N/A 
		Returns:
			a list of Entities that was created from the alerts CSV passed in 
		"""
	entList = cleanData('ioc_hunt_long.csv')
	return entList

def enrich(ent,action,randomEnrichemnts):
	"""Enriches the data by calling the required tool based on the action passed in.
	   This updates the state with what the enrichment tool returns
		   
		Args: 
			ent: the entity we are currently working with
			action: chooses which enrichment tool we will be using 
			randomEnrichments: list of random returns from tools used for mockups 
		Returns:
			the state with the updated result from the enrichemnt tool        	
	"""
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
	"""Based on how many tools the state has gotten good enrichment from decide a reward amount to 
	   give the learner. If the current state is the same as the oldState (ie. No positive move was made)
	   return a negative reward
		   
		Args: 
			state: the current state we are in 
			oldState: the state before an enrichment tool was tried 
		Returns:
			an integer reward       	
	"""
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
	return -12

def assocRandomHelper(lst,ele):
	"""Helper for the assocRandom function
		   
		Args: 
			lst: list of tuples of (alert,randomEnrichments)
			ele: randomEnrichment to match on 
		Returns:
			returns the associated randomEnrichment array to an alert field
			if it is already in the lst and None if there is no instance        	
	"""
	for i in lst:
		if np.array_equal(i[0] , ele): 
			return i[1]
	return None	

def assocRandom(ents):
	"""Makes a list of tuples of an alert mapped to a list of random enrichment results
	   Note: The list of random enrichment tools will stay the same with the same alert to ensure
	   the same actions are taken if the alert is the same
		   
		Args: 
			ents: the list of entities
		Returns:
			list of tuples of each alert mapped to its own list of random enrichments       	
	"""
	randAssoc= []
	for ent in ents:
		temp = assocRandomHelper(randAssoc,ent.state[:11])
		if temp == None:
			randAssoc.append((ent.state[:11],randoEnrich()))
		else:
			randAssoc.append((ent.state[:11],temp))
	return randAssoc


def randoEnrich():
	"""Creates a list of 1s and 0s. 1 represents if an enrichment tool returned meaningful results,
	   0 means nothing meaningful was returned
		Args: 
			N/A
		Returns:
			list of whether enrichment tools returned meaningful information        	
	"""
	randomEnrichemnts = []
	for i in range(6):
		randomEnrichemnts.append(np.random.randint(0,2))

	# while np.count_nonzero(randomEnrichemnts) < 3:
	# 	randomEnrichemnts[np.random.randint(0,6)]=1


	return randomEnrichemnts




def virusTotal():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score	

def whiteList():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score

def passiveTotal():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score

def a():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score

def b():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score

def c():
	"""Right now just returns a random 1 or 0 to represent meaningful data or no
	   In the future will return actual enrichment information from this tool 
		   
		Args: 
			N/A
		Returns:
			random 1 or 0 to show meaningful or not meaningful respectively        	
	"""
	score =np.random.randint(0,2)
	return score


def dispData(state):
	"""Prints the state 
		   
		Args: 
			N/A
		Returns:
			N/A        	
	"""
	print (state)

def initNN(inputNodes,outputNodes):
	"""initializes the neural network we use to control our Q table 
		Args: 
			inputNodes: number of input nodes for the NN
			outputNodes: number of output nodes for the NN
		Returns:
			the model for the NN         	
	"""
	model = Sequential()
	model.add(Dense(50, init='lecun_uniform', input_shape=(inputNodes,)))
	model.add(Activation('relu'))

	model.add(Dense(50, init='lecun_uniform'))
	model.add(Activation('relu'))

	model.add(Dense(outputNodes, init='lecun_uniform'))
	model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)
	return model

#----------------------------------------------------------------------------------------------------------
def incCounter(lst,ele,inc):
	for key,val in lst.items():
		if key == ele:
			if inc == True:
				lst[key]+=1 
			return lst ,lst[key]
	if inc == True:		
		lst[ele]=0
	return lst,0

def deleteQ(qChosen,qtab):
	temp=deepcopy(qtab)
	# print('temp',temp)
	for i in qChosen:
		# print(temp,i)
		temp[i]= -9999
	return temp

def runLearner(epochs,gamma,learningRate,epsilon):
	"""Runs the Q-learner with a NN as the Q table 
		Args: 
			epochs: number of runs (number of alerts we have)
			gamma: gamma associated with the Q equation
			learningRate: learning rate associated with the Q equation
			epsilon: determines how often we make a random action 

		Returns:
			nothing          	
	"""
	inputNodes = 17
	outputNodes = 6

	exportList=[]
	exportEnrichment=[]


	#initialize our Neural Network with 17 input nodes and 6 output nodes
	# model = initNN(inputNodes,outputNodes)
	#get the data as a list of entity object where each entity is an alert 
	ents = initData()
	#for now we are getting a map of each alert to a random reward list to simulate what these tools would actually return
	enrichMap=(assocRandom(ents))

	qtable = {}

	counter = {}
	#loop though how many alerts we have in the file
	for i in range(epochs):
		# print('\nGame',i)

		#keep track of number of random actions for printing
		randActions = [] 
		#keep track of the number of runs so we can determine if it was an exhastive run
		numRuns=0
		#blank array used to keep track of which enrichment tools we've used on that run 
		toolChecked =[0,0,0,0,0,0]
		#the current ent to work on 
		ent = ents[i]  
		#if the current alert should keep being investigated or we should move on to a new one
		status = 1
		#grab the randomEnrichment list that is associated with the current alert
		randomEnrichemnts =enrichMap[i][1] 
		# randomEnrichemnts=[0,1,0,1,1,1]
		qChosen=[]
		totalRuns = 0
		#keep going while we want to keep investigating this alert 
		while(status == 1):
			alertString = ''.join(str(e) for e in ent.state)
			randFlag=False
			totalRuns+=1
			
			#Run our Q function on the current state to get Q values for all possible actions
			if alertString in qtable:
				qval = qtable[alertString]
			else:
				qtable[alertString]=[0,0,0,0,0,0]
				qval = qtable[alertString]
			#decide based on epsilon whether to take a random action or chose an action from the Q table

			counter,count =incCounter(counter,''.join(str(e) for e in ent.state[:11]),False)

			# epsilon = (1/((count+1)))

			if (random.random() < epsilon): 
				action = np.random.randint(0,outputNodes)
					
			else: 				
				randFlag = True	
				temp=deleteQ(qChosen,qval)				
				action = (np.argmax(temp))				
				qChosen.append(action)
				
			#make a deepcopy of the state before it is updated with the new action        
			oldState = deepcopy(ent.state)

			#if we havent used the chosen tool yet 
			if toolChecked[action] != 1:
				#enrich the data using the chosen action
				new_state = enrich(ent, action , randomEnrichemnts)
				#get the reward associated with thaty enrichment
				reward = getReward(new_state,oldState)
				optWin = np.count_nonzero(randomEnrichemnts)

				#mark down that we've checked this tool 
				toolChecked[action]=1
				# print('Rand Actions:', randActions,'Total Runs:',totalRuns, 'Epsilon',epsilon)
				# print('Print Q table', qval,'Random Enrichment Decisions:', randomEnrichemnts)
				# print('Alert: ', ent.state[:11], 'Action:',action)
				if randFlag == True:
					randActions.append(action)
				#Get max_Q(S',a)
				newAlertString = ''.join(str(e) for e in new_state)

				if newAlertString in qtable:
					newQ = qtable[newAlertString]
				else:
					qtable[newAlertString]=[0,0,0,0,0,0]
					newQ = qtable[newAlertString]

				
				maxQ = np.max(newQ)

				# y = np.zeros((1,outputNodes))
				# y[:] = qval[:]


				# if we're in a non-terminal state
				if np.count_nonzero(ent.state[-6:]) != np.count_nonzero(randomEnrichemnts):
					# print ('Reward:',reward,'maxQ:',maxQ)
					update = learningRate * (reward + (gamma * maxQ))
				#terminal state		        
				else: 
					update = reward

				#target output
				
				# print('Update',update)

				#refit the data based on the reward gained from this run 
				qtable[alertString][action]=update
				# print('Print Q table after fit', qtable[alertString])
				# print('\n\n\n')
				#make the old state the new state
				ent.state = new_state
			#add a run 
				numRuns+=1
			
			#check if we have a sufficent reward or we've exhausted all searches and need to quit
			if np.count_nonzero(ent.state[-6:]) == np.count_nonzero(randomEnrichemnts) or numRuns > 5:		
				status = 0
				#print statements to show result
				if np.count_nonzero(ent.state[-6:]) == np.count_nonzero(randomEnrichemnts):
					counter,count =incCounter(counter,''.join(str(e) for e in ent.state[:11]),True)
					print('Policy:',ent.policy)
					print('Random Enrichment Decisions:', randomEnrichemnts)
					print('Chosen Actions:', randActions,'Epsilon:',epsilon)
					print('Print Q table', qval)
					print('Alert: ', ent.state[:11],'Random Enrichment Decisions:', randomEnrichemnts, 'Count:',count)
					if np.array_equal( [1,0,1,0,0,0,1,1,0,0,0],ent.state[:11]):
						exportList.append(ent.policy)
						exportEnrichment=randomEnrichemnts
						# print ('HIT\n\n\n')

		#keep decreasing epsilon so we can chose from the Q table more often 		
		if epsilon > 0:
			epsilon -= (1/epochs)
	j =0
	optPol=[]
	while j < len(exportEnrichment):
		if exportEnrichment[j] == 1:
			optPol.append(j)
		j+=1

	return exportList,optPol



lst,en=runLearner(659,1,.9,.9)



def createGUI(agentPolicies, correctPolicy):

    imgFiles = ['a.png',
                'b.png',
                'c.png',
                'd.png',
                'pt.png',
                'vt.png',
                'blank.png']

    root = Tk()
    actionImgs = []

    for filename in imgFiles:
        image = Image.open(filename)
        image = image.resize((100, 100), Image.ANTIALIAS) #The (250, 250) is (height, width)
        img = ImageTk.PhotoImage(image)
        actionImgs.append(img)

    def getActionImages(policy, actionImages):
        imgs = ['','','','','','']

        # for element in policy:
        #     imgs.append(actionImages[element])

        # for element in range(0, (6-len(policy))):
        #     imgs.append('')

        for element in policy:
            imgs[element] = actionImages[element]

        for i in range(0,6):
        	if imgs[i] == '':
        		imgs[i] = actionImages[6]

        return imgs

        # imgs = []

        # for element in policy:
        #     imgs.append(actionImages[element])

        # return imgs

    correctImgs = getActionImages(correctPolicy, actionImgs)

    root.configure(bg = '#525C5B')

    text1 = Label(root, text = 'Optimal Actions: ', fg = 'white', bg = '#525C5B', font = 18).grid(row = 1, column = 0)

    text2 = Label(root, text = 'Agent\'s Actions: ', fg = 'white', bg = '#525C5B', font = 18).grid(row = 2, column = 0)

    iterationNumber = Label(root, text = 'Iteration Value: ', fg = 'white', bg = '#525C5B', font = 18).grid(row = 3, column = 0)

    iterationValue = Label(root, text = '0', bg = '#525C5B', fg = 'white', font = 18, height = 5)

    iterationValue.grid(row = 3, column = 1)

    i = 1
    for image in correctImgs:
        Label(root, image = image, borderwidth = 10, bg = '#525C5B').grid(row = 1,column = i)
        i += 1

    lab0 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab0.grid(row = 2,column = 1)

    lab1 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab1.grid(row = 2,column = 2)

    lab2 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab2.grid(row = 2,column = 3)

    lab3 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab3.grid(row = 2,column = 4)

    lab4 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab4.grid(row = 2,column = 5)

    lab5 = Label(root, image = None, borderwidth = 10, bg = '#525C5B')
    lab5.grid(row = 2,column = 6)

    def getNewPolicy(agentPolicies = agentPolicies):
        if (int(iterationValue['text'])-1) < len(agentPolicies):

            return agentPolicies[int(iterationValue['text'])-1]
        else:
            return None

    def advance(agentPolicy = None, actionImgs = actionImgs):

        iterationValue['text'] = str(int(iterationValue['text'])+1)

        agentPolicy = getNewPolicy()

        if agentPolicy is None:
            time.sleep(5)
            exit()

        def getActionImages(policy, actionImages):
            imgs = ['','','','','','']

            # for element in policy:
            #     imgs.append(actionImages[element])

            # for element in range(0, (6-len(policy))):
            #     imgs.append('')

            for element in policy:
                imgs[element] = actionImages[element]

            return imgs

        agentImgs = getActionImages(agentPolicy, actionImgs)

        root.update_idletasks()

        lab0.config(image = agentImgs[0], bg = '#525C5B')
        lab0.image = agentImgs[0]

        lab1.config(image = agentImgs[1], bg = '#525C5B')
        lab1.image = agentImgs[1]

        lab2.config(image = agentImgs[2], bg = '#525C5B')
        lab2.image = agentImgs[2]

        lab3.config(image = agentImgs[3], bg = '#525C5B')
        lab3.image = agentImgs[3]

        lab4.config(image = agentImgs[4], bg = '#525C5B')
        lab4.image = agentImgs[4]

        lab5.config(image = agentImgs[5], bg = '#525C5B')
        lab5.image = agentImgs[5]

        root.update_idletasks()
        root.after(200, advance)

    root.after(200, advance)
    root.mainloop()


createGUI(lst, en)