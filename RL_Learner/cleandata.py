import random
import numpy as np
import csv
from pprint import pprint
import re
from itertools import zip_longest

file = open('IOC_Hunt_Results2_interns_answers.csv','r')

dates=[]
iocs=[]
titles=[]
levels=[]
classifs=[]
descript = []
desAns= input('Do you have a description field (y/n): ')
reader = csv.reader(file, delimiter=',')

for row in reader:
    dates.append(row[0])
    iocs.append(row[1])
    titles.append(row[2])
    levels.append(row[3])
    classifs.append(row[4])
    if desAns == 'y' or desAns == 'Y':
    	descript.append(row[5])


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
			date_split.append(False)
		else:
			date_split.append(True)
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

dates[0]='DATES'




low,med,high,undef = threatLevel(levels)
clean_data = zip_longest(dates,classifDate(dates),isIP(iocs),isURL(iocs),isHash(iocs),knownThreat(titles),levels,low,med,high,undef,classifs,descript)
outfile = open("clean_data.csv", "w",newline='')
writer = csv.writer(outfile)
writer.writerows(clean_data)
outfile.close()
