#!/usr/bin/env python
# encoding: utf-8 
""" 
@site: 
@software: PyCharm Community Edition
@file: Titanic.py
@time: 2017/12/8 22:39
这一行开始写关于本文件的说明与解释 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Yirenpy.YirenSupported import YirenSetting
from sklearn.preprocessing import OneHotEncoder
# read data
train = pd.read_csv(".\\Kaggle\\Titanic\\train.csv")
test = pd.read_csv(".\\Kaggle\\Titanic\\test.csv")
train = train.set_index("PassengerId")
test = test.set_index("PassengerId")



# data describe
print train.head()

# deal data
def deal_data(train, test):
	"""
	Variable	Definition	                        Key
	survival	Survival	                        0 = No, 1 = Yes
	pclass	    Ticket class	                    1 = 1st, 2 = 2nd, 3 = 3rd
	sex	        Sex
	Age	        Age in years
	sibsp	    # of siblings / spouses aboard the Titanic
	parch	    # of parents / children aboard the Titanic
	ticket	    Ticket number
	fare	    Passenger fare
	cabin	    Cabin number
	embarked	Port of Embarkation	                C = Cherbourg, Q = Queenstown, S = Southampton
	"""
	fulldf = [train.copy(), test.copy()]
	for df in fulldf[::-1]:

		# fill_na
		df.describe()
		df.isnull().sum()
		df.groupby("Embarked")["Embarked"].count()
		df["Cabin"] = df["Cabin"].apply(lambda x:1 if type(x)==float else 0)
		df["Embarked"] = df["Embarked"] .fillna("S")

		age_mean = df["Age"].mean()
		age_std = df["Age"].std()
		age_null_count = df["Age"].isnull().sum()
		age_fillna = np.random.randint(age_mean-2*age_std,age_mean+2*age_std,age_null_count)
		df.loc[df["Age"].isnull(),"Age"] = age_fillna

		# ont-hot
		Sex = df["Sex"].drop_duplicates().tolist()
		Sexs = map(lambda x: "Sex_" + x, Sex)
		for i,j in zip(Sex,Sexs) :
			df[j] = df["Sex"].apply(lambda x: 1 if x == i else 0)
		Embarked = df["Embarked"].drop_duplicates().tolist()
		Embarkeds = map(lambda x:"Embarked_" + x, Embarked)
		for i,j in zip(Embarked,Embarkeds) :
			df[j] = df["Embarked"].apply(lambda x: 1 if x == i else 0)

		# familysize
		df["Family_size"] = df["SibSp"] + df["Parch"] + 1
		df["Fare_average"] = df["Fare"]/df["Family_size"]

		# drop columns
		drop_columns = ["Name","Ticket","SibSp","Parch","Fare","Embarked","Sex"]
		df.drop(labels=drop_columns,axis=1,inplace=True)

	return fulldf
train, test = deal_data(train, test)

# corr analysis
colormap = plt.cm.RdBu
sns.heatmap(train.astype(float).corr(),
            linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)


# class model fit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
model = ExtraTreesClassifier()
model = RandomForestClassifier()
model = GradientBoostingClassifier()
model = AdaBoostClassifier()
model = SVC()
model = MLPClassifier(
	activation='tanh', hidden_layer_sizes=(100,50,), max_iter=2000,
	shuffle=True, solver='adam', tol=0.00001, validation_fraction=0.1)



enc = KFold(5,shuffle=True)
kfold = enc.split(train)
for index_train,index_test in kfold:
	kf_train = train.loc[index_train]
	kf_test  = train.loc[index_test]
	# 原因不明，含有包含0的样本记录
	kf_train = kf_train[kf_train.isnull().sum(axis=1)==0]
	kf_test  = kf_test [kf_test. isnull().sum(axis=1)==0]
	cols = train.columns.tolist()
	cols.remove("Survived")
	X  = np.array(kf_train[cols])
	y  = np.array(kf_train["Survived"]).reshape(-1, 1)
	X2 = np.array(kf_test[cols])
	y2 = np.array(kf_test["Survived"]).reshape(-1, 1)
	model.fit(X, y)
	print model.score(X, y),model.score(X2, y2)


# defined a customized class



