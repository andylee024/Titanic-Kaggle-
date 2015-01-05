import csv as csv
import numpy as np
import pandas as pd
import math


"""
Function Name: clean_data()
Input: data_frame object
Purpose: Cleans data and preps data frame for analysis
Status: Completed!

Notes:
I chose a specific set of features that I felt were relevant for prediction. 
This includes sex, family, embarked, age, P-class and fare price. 

My approach to cleaning the data was to force all the features to take on roughly 1-6 discrete values. 
Hence, I treated each specific feature (possibly continuous values) and discretized its values within a relevant range.  
Implementation details can be seen in the code. 

"""

def family(x): # helper function created for family size feature
	if x <=3: return 1
	else: return 0


def build_df(data):

	#Build Dataframe using titanic data
	df = pd.read_csv(data,header=0)
	
	#############################
	# Construct Relevant Features
	#############################

	# 1) Sex | females:0 , males:1
	df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	
	# 2) Family (total number of family members)
	df['Family'] = df['SibSp'] + df['Parch'] #Sibsp = no. of siblings/spouses, Parch = no. spouses/children

	#Fill Null Values
	if len(df.Family[df.Family.isnull() ]) > 0:
		df.loc[ ( df.Family.isnull() ), 'Family'] = df.Family.mode()[0]
	df['Family'] = df['Family'].map( lambda x : family(x)) #convert to binary based on whether there are 3+ family members

	# 3) Embarked - S:0, Q:1, C:2
	df.loc[ (df.Embarked.isnull()), "Embarked"] =  df.Embarked.mode()[0] #fill null values with mode
	df['Embarked'] = df['Embarked'].map( {'S':0, 'Q':1, 'C':2} ).astype(int) #convert to integer values

	# 4) Age 
	
	#Fill Null Values
	if len(df.Age[df.Age.isnull() ]) > 0:
		median_ages = np.zeros((2,3)) #Determine median ages based on sex and pclass

		for i in range(0, 2):
			for j in range(0, 3):
				median_ages[i,j] = df[ (df['Sex'] == i) & (df['Pclass'] == j+1) ]\
				['Age'].dropna().median()
		
		#Fill in null values of Age
		for i in range(0, 2):
			for j in range(0, 3):
				df.loc[ ( df.Age.isnull() ) & ( df['Sex'] == i ) & (df['Pclass'] == j+1), \
				'Age'] = median_ages[i,j]

	#Classify according to age brackets (10s, 20s, 30s, ...)
	df['Age']=df['Age'].map( lambda x: math.floor(x/10.0)*10) 


	# 5) Fare Price 	
	
	#Fill Null Values
	if len(df.Fare[df.Fare.isnull() ]) > 0:	
		median_fare = np.zeros(3) #substitute median fare based on pclass

		for i in range(3):                                              # loop 0 to 2
			median_fare[i] = df[df.Pclass == i+1 ]['Fare'].dropna().median()
		for i in range(3):                                              # loop 0 to 2
			df.loc[ (df.Fare.isnull() ) & (df.Pclass == i+1 ), 'Fare'] = median_fare[i]
	
	#Convert fares into brackets 0: $(0-10) | 1: $(10-20) | ... | 4: $(30-40) | 5: $(40+)
	for i in range(6):
		df.loc[ df.Fare >= 10*i, 'Fare_Bracket'] = i

	###############################
	# Drop Irrelevant Features
	###############################
	ids = df['PassengerId'].values #save ids	
	df = df.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'SibSp','Parch','Fare','Embarked'], axis=1)
	df = df.drop(['Age','Fare_Bracket'], axis = 1) #dropping these features oddly improves score
	return df.values, ids
