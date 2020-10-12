# ML2 Randomized Optimization

import pandas as pd
import numpy as np
import math
import time

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose_hiive

import matplotlib.pyplot as plt


from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

print ('HELLO WORLD')

from sklearn.datasets import load_iris
data = load_iris()

# PART 1: Optimizing NN weight using :
#           1. Backprop (BP)
#           2. Randomized Hill Climbing (RHC)
#           3. Simulated Annealing (SA)
#           4. A genetic Alg  (GA)
if (0):
	f = open("resultP1_BPonly.txt", "w")

	# header: SFH,popUpWidnow,SSLfinal_State,Request_URL,URL_of_Anchor,web_traffic,URL_Length,age_of_domain,having_IP_Address,Result
	col_names  = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']
	pima = pd.read_csv("PhishingData.csv", header=None, names=col_names)

	feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
	X = pima[feature_cols].to_numpy() # Features
	X = X[1:]
	y = (pima.Result).to_numpy() # Target variable
	y = y[1:]


#	sick = pd.read_csv("dataset_38_sick_cleanup.csv")
#	sick_nonan = sick.dropna()
#	X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
#	X = X_pre.to_numpy() # Features
#	y = (sick_nonan.Class).to_numpy() # Target variable

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1) # 70% training and 30% test


	n_features = X_train.shape[1]
	print (X_train.shape)
	scaler = MinMaxScaler()

	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# One hot encode target values
	one_hot = OneHotEncoder()

	y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
	y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
	print (3*n_features//2)
	print (n_features)

	x = range(1, 1000, 20)

	if (0): # Backprop from part 1
		print ('backprop')
		f.write("\n BACKPROP \n")
		t0= time.clock()

		accuracy_BP = np.zeros(len(x))
		index = 0
		for num in x:
			print (num)
			clf = MLPClassifier(max_iter = num)
			clf = clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)
			accuracy_BP[index] = metrics.accuracy_score(y_test, y_pred)
			f.write("%s %s \n" % (num, accuracy_BP[index]))
			index = index + 1

		fig = plt.figure(1)
		plt.plot(x ,accuracy_BP)
		plt.ylabel('Accuracy')
		plt.xlabel('Iterations ')
		plt.title('SicknessDetection - NN Backprop - Accuracy vs Iterations ')
		plt.grid()
		fig.tight_layout()
		plt.legend()
		fig.savefig('P1_BP.png')

		t1 = time.clock() - t0
		f.write("Runtime %s seconds \n" % (t1))
		print('back prob done!')

	if (0): # RHC
		print('\n random_hill_climb')
		f.write("\n RHC \n")
		t0= time.clock()
		accuracy_RHC1 = np.zeros(len(x))
		accuracy_RHC2 = np.zeros(len(x))
		accuracy_RHC3 = np.zeros(len(x))
		accuracy_RHC4 = np.zeros(len(x))
		index = 0
		for num in x:
			print (num)
			rhcmodel1 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='random_hill_climb',
										restarts = 0,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			rhcmodel2 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='random_hill_climb',
										restarts = 5,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			rhcmodel3 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='random_hill_climb',
										restarts = 10,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			rhcmodel4 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='random_hill_climb',
										restarts = 15,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			rhcmodel1.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel1.predict(X_train_scaled)
			accuracy_RHC1[index] = accuracy_score(y_train_hot, y_train_pred)

			rhcmodel2.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel2.predict(X_train_scaled)
			accuracy_RHC2[index] = accuracy_score(y_train_hot, y_train_pred)

			rhcmodel3.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel3.predict(X_train_scaled)
			accuracy_RHC3[index] = accuracy_score(y_train_hot, y_train_pred)

			rhcmodel4.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel4.predict(X_train_scaled)
			accuracy_RHC4[index] = accuracy_score(y_train_hot, y_train_pred)

			f.write("%s %s %s %s %s \n" % (num, accuracy_RHC1[index], accuracy_RHC2[index], accuracy_RHC3[index], accuracy_RHC4[index]))

			index = index + 1


		fig = plt.figure(2)
		plt.plot(x ,accuracy_RHC1, label="restarts = 0")
		plt.plot(x ,accuracy_RHC2, label="restarts = 5")
		plt.plot(x ,accuracy_RHC3, label="restarts = 10")
		plt.plot(x ,accuracy_RHC4, label="restarts = 15")
		plt.ylabel('Accuracy')
		plt.xlabel('Iterations ')
		plt.title('SicknessDetection - Random Hill Climb - Accuracy vs Iterations ')
		plt.grid()
		plt.legend()
		fig.tight_layout()
		fig.savefig('P1_RHC.png')

		t1 = time.clock() - t0
		f.write("Runtime %s seconds \n" % (t1))

		print('rhc done!')



	if (0): # SA
		print('\n sa')
		f.write("\n SA \n")
		t0= time.clock()
		accuracy_SA1 = np.zeros(len(x))
		accuracy_SA2 = np.zeros(len(x))
		accuracy_SA3 = np.zeros(len(x))
		index = 0
		for num in x:
			print (num)
			samodel1 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='simulated_annealing',
										schedule = mlrose.GeomDecay(),
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)
			samodel2 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										schedule = mlrose.ArithDecay(),
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)
			samodel3 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										schedule = mlrose.ExpDecay(),
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			samodel1.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel1.predict(X_train_scaled)
			accuracy_SA1[index] = accuracy_score(y_train_hot, y_train_pred)

			samodel2.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel2.predict(X_train_scaled)
			accuracy_SA2[index] = accuracy_score(y_train_hot, y_train_pred)

			samodel3.fit(X_train_scaled, y_train_hot)
			y_train_pred = rhcmodel3.predict(X_train_scaled)
			accuracy_SA3[index] = accuracy_score(y_train_hot, y_train_pred)

			f.write("%s %s %s %s \n" % (num, accuracy_SA1[index], accuracy_SA2[index], accuracy_SA3[index]))

			index = index + 1

		fig = plt.figure(3)
		plt.plot(x ,accuracy_SA1, label="schedule = GeomDecay()")
		plt.plot(x ,accuracy_SA2, label="schedule = ArithDecay()")
		plt.plot(x ,accuracy_SA3, label="schedule = ExpDecay()")
		plt.ylabel('Accuracy')
		plt.xlabel('Iterations ')
		plt.title('SicknessDetection - Simulated Annealing Accuracy vs Iterations ')
		plt.grid()
		plt.legend()
		fig.tight_layout()
		fig.savefig('P1_SA.png')

		t1 = time.clock() - t0
		f.write("Runtime %s seconds \n" % (t1))

		print('sa done!')

	if (0): # GA
		print('\n ga')
		f.write("\n GA \n")

		t0= time.clock()
		accuracy_GA1 = np.zeros(len(x))
		accuracy_GA2 = np.zeros(len(x))
		accuracy_GA3 = np.zeros(len(x))
		accuracy_GA4 = np.zeros(len(x))
		index = 0
		for num in x:
			print (num)
			gamodel1 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='genetic_alg',
										mutation_prob = 0.05,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			gamodel2 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='genetic_alg',
										mutation_prob = 0.10,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			gamodel3 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='genetic_alg',
										mutation_prob = 0.15,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			gamodel4 = mlrose_hiive.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid',
										algorithm ='genetic_alg',
										mutation_prob = 0.20,
										max_iters = num, bias = False, is_classifier = True,
										learning_rate=0.001, early_stopping = True,
										clip_max = 5, max_attempts = 50)

			gamodel1.fit(X_train_scaled, y_train_hot)
			y_train_pred = gamodel1.predict(X_train_scaled)
			accuracy_GA1[index] = accuracy_score(y_train_hot, y_train_pred)

			gamodel2.fit(X_train_scaled, y_train_hot)
			y_train_pred = gamodel2.predict(X_train_scaled)
			accuracy_GA2[index] = accuracy_score(y_train_hot, y_train_pred)

			gamodel3.fit(X_train_scaled, y_train_hot)
			y_train_pred = gamodel3.predict(X_train_scaled)
			accuracy_GA3[index] = accuracy_score(y_train_hot, y_train_pred)

			gamodel4.fit(X_train_scaled, y_train_hot)
			y_train_pred = gamodel4.predict(X_train_scaled)
			accuracy_GA4[index] = accuracy_score(y_train_hot, y_train_pred)

			f.write("%s %s %s %s %s \n" % (num, accuracy_GA1[index], accuracy_GA2[index], accuracy_GA3[index], accuracy_GA4[index]))

			index = index + 1

		fig = plt.figure(4)
		plt.plot(x ,accuracy_GA1, label="mutation_prob = 0.05")
		plt.plot(x ,accuracy_GA2, label="mutation_prob = 0.10")
		plt.plot(x ,accuracy_GA3, label="mutation_prob = 0.15")
		plt.plot(x ,accuracy_GA4, label="mutation_prob = 0.20")
		plt.ylabel('Accuracy')
		plt.xlabel('Iterations ')
		plt.title('SicknessDetection - Genetic Algorithm Accuracy vs Iterations ')
		plt.grid()
		plt.legend()
		fig.tight_layout()
		fig.savefig('P1_GA.png')


		t1 = time.clock() - t0
		f.write("Runtime %s seconds \n" % (t1))

		print('ga done!')

	f.close()

if (0): # plotting combined. got the data from the log in the same folder
	x = [1   ,
		21  ,
		41  ,
		61  ,
		81  ,
		101 ,
		121 ,
		141 ,
		161 ,
		181 ,
		201 ,
		221 ,
		241 ,
		261 ,
		281 ,
		301 ,
		321 ,
		341 ,
		361 ,
		381 ,
		401 ,
		421 ,
		441 ,
		461 ,
		481 ,
		501 ,
		521 ,
		541 ,
		561 ,
		581 ,
		601 ,
		621 ,
		641 ,
		661 ,
		681 ,
		701 ,
		721 ,
		741 ,
		761 ,
		781 ,
		801 ,
		821 ,
		841 ,
		861 ,
		881 ,
		901 ,
		921 ,
		941 ,
		961 ,
		981]
	y1 = [1] * 4
	y2 = [2] * 4
	y3 = [3] * 4
	y4 = [4] * 4

	y_BP =[0.6330049261083743,
			0.8251231527093597,
			0.8300492610837439,
			0.8423645320197044,
			0.8448275862068966,
			0.854679802955665 ,
			0.8669950738916257,
			0.8645320197044335,
			0.8694581280788177,
			0.8793103448275862,
			0.8891625615763546,
			0.896551724137931 ,
			0.8916256157635468,
			0.9014778325123153,
			0.8916256157635468,
			0.8990147783251231,
			0.8817733990147784,
			0.8916256157635468,
			0.8990147783251231,
			0.8990147783251231,
			0.8866995073891626,
			0.9014778325123153,
			0.8940886699507389,
			0.8940886699507389,
			0.8891625615763546,
			0.9014778325123153,
			0.896551724137931 ,
			0.8940886699507389,
			0.8891625615763546,
			0.896551724137931 ,
			0.9014778325123153,
			0.8990147783251231,
			0.8940886699507389,
			0.8940886699507389,
			0.896551724137931 ,
			0.896551724137931 ,
			0.8866995073891626,
			0.8866995073891626,
			0.9039408866995073,
			0.8916256157635468,
			0.8940886699507389,
			0.8891625615763546,
			0.9014778325123153,
			0.8990147783251231,
			0.896551724137931 ,
			0.8916256157635468,
			0.8990147783251231,
			0.8891625615763546,
			0.8940886699507389,
			0.8940886699507389 ]

	y_RHC =[0.90,
			0.11,
			0.36,
			0.21,
			0.08,
			0.92,
			0.09,
			0.15,
			0.92,
			0.08,
			0.08,
			0.30,
			0.58,
			0.80,
			0.09,
			0.91,
			0.31,
			0.92,
			0.65,
			0.09,
			0.08,
			0.91,
			0.90,
			0.08,
			0.92,
			0.14,
			0.08,
			0.09,
			0.08,
			0.92,
			0.10,
			0.53,
			0.91,
			0.08,
			0.79,
			0.08,
			0.08,
			0.91,
			0.08,
			0.84,
			0.60,
			0.08,
			0.08,
			0.90,
			0.49,
			0.92,
			0.92,
			0.91,
			0.08,
			0.92]

	y_SA =[0.92,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92 ,
			0.92]

	y_GA = [0.8718918,
			0.8951351,
			0.9113513,
			0.9086486,
			0.9091891,
			0.9183783,
			0.9016216,
			0.9178378,
			0.9156756,
			0.9021621,
			0.9172972,
			0.9243243,
			0.9151351,
			0.9075675,
			0.9075675,
			0.9259459,
			0.9243243,
			0.9291891,
			0.9097297,
			0.9259459,
			0.9086486,
			0.9200000,
			0.9227027,
			0.9227027,
			0.9351351,
			0.9205405,
			0.9243243,
			0.9378378,
			0.9329729,
			0.9113513,
			0.9194594,
			0.9232432,
			0.9335135,
			0.9086486,
			0.9059459,
			0.9248648,
			0.9043243,
			0.9118918,
			0.9086486,
			0.9254054,
			0.9189189,
			0.8962162,
			0.9189189,
			0.9205405,
			0.9081081,
			0.9059459,
			0.9216216,
			0.9194594,
			0.9243243,
			0.9027027]

	print (len(x))
	fig  = plt.figure(10)
	plt.plot(x ,y_BP, label="Best Backprop Accuracy")
	plt.plot(x ,y_RHC, label="Best RHC Accuracy")
	plt.plot(x ,y_SA, label="Best SA Accuracy")
	plt.plot(x ,y_GA, label="Best GA Accuracy")
	plt.ylabel('Accuracy')
	plt.xlabel('Iterations ')
	plt.title('SicknessDetection - Best Accuracy from 4 methods vs Iterations ')
	plt.grid()
	plt.legend()
	fig.savefig('P1_combined.png')



if (0): # plot runtime got it from the log
	x =['BP', 'RHC', 'SA', 'GA']
	y = [87.63, round(2977.2727868/4,1), round(445.03771549999965/3,1), round(11561.606253599999/4,1)]
	fig, ax = plt.subplots()
	ax.bar(x, y, width = 0.5, color='r')
	plt.ylabel('Runtime (s)')
	plt.xlabel('Methods')
	ax.set_xticklabels(x)
	ax.set_ylim([0,4000])
	for i in range(len(y)):#
		plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')
#	plt.grid()
	ax.yaxis.grid()
	plt.title('SicknessDetection - NN Runtime (seconds) Comparison ')

if (0): # Best accuracy
	x =['BP', 'RHC', 'SA', 'GA']
	y = [90.12, round(0.925346523*100,2), round(0.92*100,2), round(0.9378378*100,2)]
	fig, ax = plt.subplots()
	ax.bar(x, y, width = 0.5)
	ax.set_ylim([80,100])
	plt.ylabel('Accuracy (%)')
	plt.xlabel('Methods')
	ax.set_xticklabels(x)
	for i in range(len(y)):#
		plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')
#	plt.grid()
	ax.yaxis.grid()
	plt.title('SicknessDetection - NN Best Accuracy (%) Comparison ')








plt.show()
