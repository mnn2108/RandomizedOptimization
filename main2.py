# main2.py

# Part 2 to run in parallel

import pandas as pd
import numpy as np
import math
import time
import random

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


# ************************************************

# PART 2: Apply RHC, SA, GA, and MIMIC to 3 optimization problems:
#        1. Flip Flop Problem (FFP)   => MIMIC is best, Ga is worst
#        2. Continuous Peaks Problem (CPP) or FourPeak   => GA is best
#        3. Travelling Salesman Problem (TSP)	   =>  SA is best, MIMIC is worst
#        4. Count Ones   => SA and RHC is best
#        5. Knapsack   => MIMIC is best

if(1):
	print ('HELLO PART 2')
	x = range(1, 1000, 20)


	if (0):	# Continuous Peaks Problem (CPP)
		f = open("resultP2_CPP.txt", "a")
		f.write("\n resultP2_CPP \n")
		print ('\nProblem 2: Continuous Peak (CPP)')
		problem_size = 50
		peaks_fit = mlrose.ContinuousPeaks(t_pct=.1)
		prob_size_int = 100
		problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=peaks_fit, maximize=True, max_val=2)
		cpeaks_state_gen = lambda: np.random.randint(2, size=prob_size_int)
		init_state = cpeaks_state_gen()
		x = range(1, 1000, 20)
		n = 1000

		if (0):
			print('\n random_hill_climb')
			f.write("\n RHC \n")
			t0= time.clock()
			accuracy_RHC1 = np.zeros(len(x))
			accuracy_RHC2 = np.zeros(len(x))
			accuracy_RHC3 = np.zeros(len(x))
			accuracy_RHC4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fit_array_sa1 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=0,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness2, fit_array_sa2 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=5,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness3, fit_array_sa3 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=10,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness4, fit_array_sa4 = mlrose.random_hill_climb(problem,
																			max_attempts=10,
																			max_iters=n,
																			restarts=15,
																			init_state=None,
																			curve=False,
																			random_state=None)

				accuracy_RHC1[index] = accuracy_score(init_state, best_fitness1)
				accuracy_RHC2[index] = accuracy_score(init_state, best_fitness2)
				accuracy_RHC3[index] = accuracy_score(init_state, best_fitness3)
				accuracy_RHC4[index] = accuracy_score(init_state, best_fitness4)
				fitness_score1[index] = fit_array_sa1
				fitness_score2[index] = fit_array_sa2
				fitness_score3[index] = fit_array_sa3
				fitness_score4[index] = fit_array_sa4
				f.write("%s %s %s %s %s,   %s %s %s %s \n" % (num, accuracy_RHC1[index], accuracy_RHC2[index], accuracy_RHC3[index], accuracy_RHC4[index], fit_array_sa1, fit_array_sa2, fit_array_sa3, fit_array_sa4))

				index = index + 1

			fig = plt.figure(2)
			plt.plot(x ,fitness_score1, label="restarts = 0")
			plt.plot(x ,fitness_score2, label="restarts = 5")
			plt.plot(x ,fitness_score3, label="restarts = 10")
			plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Firness')
			plt.xlabel('Iterations ')
			plt.title('Problem: CPP - Random Hill Climb - Fitness vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_RHC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('RHC DONE. Runtime = ', t1)


		if (0):
			print('\n SA')
			f.write("\n SA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.GeomDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)
				best_fitness2, fscore2 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ArithDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness3, fscore3 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness4, fscore4 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)
				accuracy1[index] = accuracy_score(init_state, best_fitness1)
				accuracy2[index] = accuracy_score(init_state, best_fitness2)
				accuracy3[index] = accuracy_score(init_state, best_fitness3)
				accuracy4[index] = accuracy_score(init_state, best_fitness4)
				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				f.write("%s %s %s %s,  %s %s %s \n" % (num, accuracy1[index], accuracy2[index], accuracy3[index],  fscore1, fscore2, fscore3))

				index = index + 1

			fig = plt.figure(2)
			plt.plot(x ,accuracy1, label="schedule = GeomDecay")
			plt.plot(x ,accuracy2, label="schedule = ArithDecay")
			plt.plot(x ,accuracy3, label="schedule = ExpDecay")
		#	plt.plot(x ,accuracy4, label="restarts = 15")
			plt.ylabel('Accuracy')
			plt.xlabel('Iterations ')
			plt.title('Problem: CPP - Simulated Annealing  - Accuracy vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_SA_accuracy.png')

			fig = plt.figure(3)
			plt.plot(x ,fitness_score1, label="schedule = GeomDecay")
			plt.plot(x ,fitness_score2, label="schedule = ArithDecay")
			plt.plot(x ,fitness_score3, label="schedule = ExpDecay")
		#	plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: CPP - Simulated Annealing - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_SA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('SA DONE. Runtime = ', t1)




		if (0):


			print('\n GA')
			f.write("\n GA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.05,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)
				best_fitness2, fscore2 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness3, fscore3 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.15,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness4, fscore4 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.20,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)


				accuracy1[index] = accuracy_score(init_state, best_fitness1)
				accuracy2[index] = accuracy_score(init_state, best_fitness2)
				accuracy3[index] = accuracy_score(init_state, best_fitness3)
				accuracy4[index] = accuracy_score(init_state, best_fitness4)
				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				f.write("%s %s %s %s %s,  %s %s %s %s \n" % (num, accuracy1[index], accuracy2[index], accuracy3[index], accuracy4[index], fscore1, fscore2, fscore3, fscore4))

				index = index + 1


			fig = plt.figure(3)
			plt.plot(x ,fitness_score1, label="mutation_prob = 0.05")
			plt.plot(x ,fitness_score2, label="mutation_prob = 0.10")
			plt.plot(x ,fitness_score3, label="mutation_prob = 0.15")
			plt.plot(x ,fitness_score4, label="mutation_prob = 0.20")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: CPP - Generic Algorithm  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_GA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('GA DONE. Runtime = ', t1)






		if (1):

			print('\n MIMIC')
			f.write("\n MIMIC \n")
			t0= time.clock()
			x = range(1, 1000, 100)
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			fitness_score5 = np.zeros(len(x))
			fitness_score6 = np.zeros(len(x))
			index = 0

			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.mimic(problem,
																pop_size=100,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness2, fscore2 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness3, fscore3 = mlrose.mimic(problem,
																pop_size=300,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness4, fscore4 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness5, fscore5 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness6, fscore6 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.3,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				fitness_score5[index] = fscore3
				fitness_score6[index] = fscore4
				f.write("%s   %s %s %s %s %s %s \n" % (num, fscore1, fscore2, fscore3, fscore4, fscore5, fscore6))

				index = index + 1


			fig = plt.figure(3)
			plt.plot(x ,fitness_score1, label="pop_size=100")
			plt.plot(x ,fitness_score2, label="pop_size=200")
			plt.plot(x ,fitness_score3, label="pop_size=300")
			plt.plot(x ,fitness_score4, label="keep_pct=0.1")
			plt.plot(x ,fitness_score5, label="keep_pct=0.2")
			plt.plot(x ,fitness_score6, label="keep_pct=0.3")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: CPP - MIMIC  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_MIMIC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('MIMIC DONE. Runtime = ', t1)







	if (0):	# Travel Saleman Problem TSP
		f = open("resultP2_TSP.txt", "a")
		f.write("\n resultP2_TSP \n")
		print ('\nProblem : Travel Saleman Problem TSP')


		prob_size_int = 20

		tsp_state_gen = lambda: np.random.choice(prob_size_int, prob_size_int, replace=False)
		print (tsp_state_gen)
		init_state = np.arange(prob_size_int)
		np.random.shuffle(init_state)
		print (init_state)

		random.seed(30) # to regenerate the problem
		coords_list = []
		for m in range(prob_size_int):
			coords_list.append((random.randint(0, 100), random.randint(0, 100)))
		print (coords_list)

		problem = mlrose.TSPOpt(prob_size_int, maximize=False, coords=coords_list)
		print (problem)

		x = range(1, 1000, 20)
		n = 1000

		if (0): # RHC
			print('\n random_hill_climb')
			f.write("\n RHC \n")
			t0= time.clock()
			accuracy_RHC1 = np.zeros(len(x))
			accuracy_RHC2 = np.zeros(len(x))
			accuracy_RHC3 = np.zeros(len(x))
			accuracy_RHC4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fit_array_sa1 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=0,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness2, fit_array_sa2 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=5,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness3, fit_array_sa3 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=10,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness4, fit_array_sa4 = mlrose.random_hill_climb(problem,
																			max_attempts=10,
																			max_iters=n,
																			restarts=15,
																			init_state=None,
																			curve=False,
																			random_state=None)


				fitness_score1[index] = fit_array_sa1
				fitness_score2[index] = fit_array_sa2
				fitness_score3[index] = fit_array_sa3
				fitness_score4[index] = fit_array_sa4
				f.write("%s  %s %s %s %s \n" % (num,  fit_array_sa1, fit_array_sa2, fit_array_sa3, fit_array_sa4))

				index = index + 1

			fig = plt.figure(1)
			plt.plot(x ,fitness_score1, label="restarts = 0")
			plt.plot(x ,fitness_score2, label="restarts = 5")
			plt.plot(x ,fitness_score3, label="restarts = 10")
			plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: TSP - Random Hill Climb - Fitness vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_TSP_RHC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('RHC DONE. Runtime = ', t1)


		if (0): # SA
			print('\n SA')
			f.write("\n SA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.GeomDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)
				best_fitness2, fscore2 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ArithDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness3, fscore3 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness4, fscore4 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
		#		fitness_score4[index] = fscore4
				f.write("%s   %s %s %s \n" % (num,  fscore1, fscore2, fscore3))

				index = index + 1


			fig = plt.figure(2)
			plt.plot(x ,fitness_score1, label="schedule = GeomDecay")
			plt.plot(x ,fitness_score2, label="schedule = ArithDecay")
			plt.plot(x ,fitness_score3, label="schedule = ExpDecay")
		#	plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: TSP - Simulated Annealing - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_TSP_SA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('SA DONE. Runtime = ', t1)


		if (0): # GA


			print('\n GA')
			f.write("\n GA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.05,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)
				best_fitness2, fscore2 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness3, fscore3 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.15,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness4, fscore4 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.20,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)



				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				f.write("%s   %s %s %s %s \n" % (num,  fscore1, fscore2, fscore3, fscore4))

				index = index + 1


			fig = plt.figure(3)
			plt.plot(x ,fitness_score1, label="mutation_prob = 0.05")
			plt.plot(x ,fitness_score2, label="mutation_prob = 0.10")
			plt.plot(x ,fitness_score3, label="mutation_prob = 0.15")
			plt.plot(x ,fitness_score4, label="mutation_prob = 0.20")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: TSP - Generic Algorithm  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_TSP_GA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('GA DONE. Runtime = ', t1)



		if (0):	# MIMIC

			print('\n MIMIC')
			f.write("\n MIMIC \n")
			t0= time.clock()

			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			fitness_score5 = np.zeros(len(x))
			fitness_score6 = np.zeros(len(x))
			index = 0
			x = range(100, 1001, 100)
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.mimic(problem,
													  pop_size=100,
													  keep_pct=0.2,
													  max_attempts=10,
													  max_iters=n,
													  curve=False,
													  random_state=None,
													  fast_mimic=False)

				best_fitness2, fscore2 = mlrose.mimic(problem,
													  pop_size=200,
													  keep_pct=0.2,
													  max_attempts=10,
													  max_iters=n,
													  curve=False,
													  random_state=None,
													  fast_mimic=False)

				best_fitness3, fscore3 = mlrose.mimic(problem,
													  pop_size=300,
													  keep_pct=0.2,
													  max_attempts=10,
													  max_iters=n,
													  curve=False,
													  random_state=None,
													  fast_mimic=False)

				best_fitness4, fscore4 = mlrose.mimic(problem,
													   pop_size=200,
													   keep_pct=0.1,
													   max_attempts=10,
													   max_iters=n,
													   curve=False,
													   random_state=None,
													   fast_mimic=False)

				best_fitness5, fscore5 = mlrose.mimic(problem,
													   pop_size=200,
													   keep_pct=0.2,
													   max_attempts=10,
													   max_iters=n,
													   curve=False,
													   random_state=None,
													   fast_mimic=False)

				best_fitness6, fscore6 = mlrose.mimic(problem,
													   pop_size=200,
													   keep_pct=0.3,
													   max_attempts=10,
													   max_iters=n,
													   curve=False,
													   random_state=None,
													   fast_mimic=False)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				fitness_score5[index] = fscore3
				fitness_score6[index] = fscore4
				f.write("%s   %s %s %s %s %s %s \n" % (num, fscore1, fscore2, fscore3, fscore4, fscore5, fscore6))

				index = index + 1


			fig = plt.figure(4)
			plt.plot(x ,fitness_score1, label="pop_size=100")
			plt.plot(x ,fitness_score2, label="pop_size=200")
			plt.plot(x ,fitness_score3, label="pop_size=300")
			plt.plot(x ,fitness_score4, label="keep_pct=0.1")
			plt.plot(x ,fitness_score3, label="keep_pct=0.2")
			plt.plot(x ,fitness_score4, label="keep_pct=0.3")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: TSP - MIMIC  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_TSP_MIMIC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('MIMIC DONE. Runtime = ', t1)


	if (0):	# Knapsack KSP
		f = open("resultP2_KSP.txt", "a")
		f.write("\n resultP2_KSP \n")
		print ('\nProblem : Knapsack KSP')


		problem_size = int(50)

		weights = [int(np.random.randint(1, problem_size/2)) for _ in range(problem_size)]
		values = [int(np.random.randint(1, problem_size/2)) for _ in range(problem_size)]
		knapsack_fit = mlrose.Knapsack(weights, values)
		flop_state_gen = lambda: np.random.randint(0, 1, size=problem_size)
		init_state = flop_state_gen()
		problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=knapsack_fit, maximize=True, max_val=2)



		x = range(1, 1000, 20)
		n = 1000

		if (0): # RHC
			print('\n random_hill_climb')
			f.write("\n RHC \n")
			t0= time.clock()
			accuracy_RHC1 = np.zeros(len(x))
			accuracy_RHC2 = np.zeros(len(x))
			accuracy_RHC3 = np.zeros(len(x))
			accuracy_RHC4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fit_array_sa1 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=0,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness2, fit_array_sa2 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=5,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness3, fit_array_sa3 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=10,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness4, fit_array_sa4 = mlrose.random_hill_climb(problem,
																			max_attempts=10,
																			max_iters=n,
																			restarts=15,
																			init_state=None,
																			curve=False,
																			random_state=None)


				fitness_score1[index] = fit_array_sa1
				fitness_score2[index] = fit_array_sa2
				fitness_score3[index] = fit_array_sa3
				fitness_score4[index] = fit_array_sa4
				f.write("%s  %s %s %s %s \n" % (num,  fit_array_sa1, fit_array_sa2, fit_array_sa3, fit_array_sa4))

				index = index + 1

			fig = plt.figure(31)
			plt.plot(x ,fitness_score1, label="restarts = 0")
			plt.plot(x ,fitness_score2, label="restarts = 5")
			plt.plot(x ,fitness_score3, label="restarts = 10")
			plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: KSP - Random Hill Climb - Fitness vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_KSP_RHC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('RHC DONE. Runtime = ', t1)


		if (0): # SA
			print('\n SA')
			f.write("\n SA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.GeomDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)
				best_fitness2, fscore2 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ArithDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness3, fscore3 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness4, fscore4 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
		#		fitness_score4[index] = fscore4
				f.write("%s   %s %s %s \n" % (num,  fscore1, fscore2, fscore3))

				index = index + 1


			fig = plt.figure(32)
			plt.plot(x ,fitness_score1, label="schedule = GeomDecay")
			plt.plot(x ,fitness_score2, label="schedule = ArithDecay")
			plt.plot(x ,fitness_score3, label="schedule = ExpDecay")
		#	plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: KSP - Simulated Annealing - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_KSP_SA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('SA DONE. Runtime = ', t1)


		if (0): # GA


			print('\n GA')
			f.write("\n GA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.05,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)
				best_fitness2, fscore2 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness3, fscore3 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.15,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness4, fscore4 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.20,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)



				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				f.write("%s   %s %s %s %s \n" % (num,  fscore1, fscore2, fscore3, fscore4))

				index = index + 1


			fig = plt.figure(33)
			plt.plot(x ,fitness_score1, label="mutation_prob = 0.05")
			plt.plot(x ,fitness_score2, label="mutation_prob = 0.10")
			plt.plot(x ,fitness_score3, label="mutation_prob = 0.15")
			plt.plot(x ,fitness_score4, label="mutation_prob = 0.20")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: KSP - Generic Algorithm  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_KSP_GA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('GA DONE. Runtime = ', t1)


		if (1):	# MIMIC

			print('\n MIMIC')
			f.write("\n MIMIC \n")
			t0= time.clock()

			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			fitness_score5 = np.zeros(len(x))
			fitness_score6 = np.zeros(len(x))
			index = 0

			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.mimic(problem,
																pop_size=100,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness2, fscore2 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness3, fscore3 = mlrose.mimic(problem,
																pop_size=300,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness4, fscore4 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness5, fscore5 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness6, fscore6 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.3,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				fitness_score5[index] = fscore3
				fitness_score6[index] = fscore4
				f.write("%s   %s %s %s %s %s %s \n" % (num, fscore1, fscore2, fscore3, fscore4, fscore5, fscore6))

				index = index + 1


			fig = plt.figure(34)
			plt.plot(x ,fitness_score1, label="pop_size=100")
			plt.plot(x ,fitness_score2, label="pop_size=200")
			plt.plot(x ,fitness_score3, label="pop_size=300")
			plt.plot(x ,fitness_score4, label="keep_pct=0.1")
			plt.plot(x ,fitness_score3, label="keep_pct=0.2")
			plt.plot(x ,fitness_score4, label="keep_pct=0.3")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: KSP - MIMIC  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_KSP_MIMIC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('MIMIC DONE. Runtime = ', t1)







	if (1):	# Flip Flop Problem FFP
		f = open("resultP2_FFP.txt", "a")
		f.write("\n resultP2_FFP \n")
		print ('\nProblem : Flip Flop Problem FFP')


		prob_size_int = 20

		ffp_fitness = mlrose.FlipFlop()
		problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=ffp_fitness, maximize=True, max_val=2)
		flop_state_gen = lambda: np.random.randint(2, size=prob_size_int)
		init_state = flop_state_gen()



		x = range(1, 1000, 20)
		n = 1000

		if (0): # RHC
			print('\n random_hill_climb')
			f.write("\n RHC \n")
			t0= time.clock()
			accuracy_RHC1 = np.zeros(len(x))
			accuracy_RHC2 = np.zeros(len(x))
			accuracy_RHC3 = np.zeros(len(x))
			accuracy_RHC4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fit_array_sa1 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=0,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness2, fit_array_sa2 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=5,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness3, fit_array_sa3 = mlrose.random_hill_climb(problem,
																		max_attempts=10,
																		max_iters=n,
																		restarts=10,
																		init_state=None,
																		curve=False,
																		random_state=None)
				best_fitness4, fit_array_sa4 = mlrose.random_hill_climb(problem,
																			max_attempts=10,
																			max_iters=n,
																			restarts=15,
																			init_state=None,
																			curve=False,
																			random_state=None)


				fitness_score1[index] = fit_array_sa1
				fitness_score2[index] = fit_array_sa2
				fitness_score3[index] = fit_array_sa3
				fitness_score4[index] = fit_array_sa4
				f.write("%s  %s %s %s %s \n" % (num,  fit_array_sa1, fit_array_sa2, fit_array_sa3, fit_array_sa4))

				index = index + 1

			fig = plt.figure(31)
			plt.plot(x ,fitness_score1, label="restarts = 0")
			plt.plot(x ,fitness_score2, label="restarts = 5")
			plt.plot(x ,fitness_score3, label="restarts = 10")
			plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: FFP - Random Hill Climb - Fitness vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_CPP_FFP_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('RHC DONE. Runtime = ', t1)


		if (0): # SA
			print('\n SA')
			f.write("\n SA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.GeomDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)
				best_fitness2, fscore2 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ArithDecay(),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness3, fscore3 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				best_fitness4, fscore4 = mlrose.simulated_annealing(problem,
																		schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
																		min_temp=.01),
																		max_attempts=50,
																		init_state=init_state,
																		max_iters=n)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
		#		fitness_score4[index] = fscore4
				f.write("%s   %s %s %s \n" % (num,  fscore1, fscore2, fscore3))

				index = index + 1


			fig = plt.figure(32)
			plt.plot(x ,fitness_score1, label="schedule = GeomDecay")
			plt.plot(x ,fitness_score2, label="schedule = ArithDecay")
			plt.plot(x ,fitness_score3, label="schedule = ExpDecay")
		#	plt.plot(x ,fitness_score4, label="restarts = 15")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: FFP - Simulated Annealing - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_FFP_SA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('SA DONE. Runtime = ', t1)


		if (0): # GA


			print('\n GA')
			f.write("\n GA \n")
			t0= time.clock()
			accuracy1 = np.zeros(len(x))
			accuracy2 = np.zeros(len(x))
			accuracy3 = np.zeros(len(x))
			accuracy4 = np.zeros(len(x))
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			index = 0
			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.05,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)
				best_fitness2, fscore2 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness3, fscore3 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.15,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)

				best_fitness4, fscore4 = mlrose.genetic_alg(problem,
																pop_size=200,
																mutation_prob=0.20,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None)



				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				f.write("%s   %s %s %s %s \n" % (num,  fscore1, fscore2, fscore3, fscore4))

				index = index + 1


			fig = plt.figure(33)
			plt.plot(x ,fitness_score1, label="mutation_prob = 0.05")
			plt.plot(x ,fitness_score2, label="mutation_prob = 0.10")
			plt.plot(x ,fitness_score3, label="mutation_prob = 0.15")
			plt.plot(x ,fitness_score4, label="mutation_prob = 0.20")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: FFP - Generic Algorithm  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_FFP_GA_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('GA DONE. Runtime = ', t1)


		if (1):	# MIMIC

			print('\n MIMIC')
			f.write("\n MIMIC \n")
			t0= time.clock()
	#		x = range(1, 1000, 100)
			fitness_score1 = np.zeros(len(x))
			fitness_score2 = np.zeros(len(x))
			fitness_score3 = np.zeros(len(x))
			fitness_score4 = np.zeros(len(x))
			fitness_score5 = np.zeros(len(x))
			fitness_score6 = np.zeros(len(x))
			index = 0

			for num in x:
				print (num)
				best_fitness1, fscore1 = mlrose.mimic(problem,
																pop_size=100,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness2, fscore2 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness3, fscore3 = mlrose.mimic(problem,
																pop_size=300,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness4, fscore4 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.1,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness5, fscore5 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.2,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				best_fitness6, fscore6 = mlrose.mimic(problem,
																pop_size=200,
																keep_pct=0.3,
																max_attempts=10,
																max_iters=n,
																curve=False,
																random_state=None,
																fast_mimic=False)

				fitness_score1[index] = fscore1
				fitness_score2[index] = fscore2
				fitness_score3[index] = fscore3
				fitness_score4[index] = fscore4
				fitness_score5[index] = fscore3
				fitness_score6[index] = fscore4
				f.write("%s   %s %s %s %s %s %s \n" % (num, fscore1, fscore2, fscore3, fscore4, fscore5, fscore6))

				index = index + 1


			fig = plt.figure(34)
			plt.plot(x ,fitness_score1, label="pop_size=100")
			plt.plot(x ,fitness_score2, label="pop_size=200")
			plt.plot(x ,fitness_score3, label="pop_size=300")
			plt.plot(x ,fitness_score4, label="keep_pct=0.1")
			plt.plot(x ,fitness_score3, label="keep_pct=0.2")
			plt.plot(x ,fitness_score4, label="keep_pct=0.3")
			plt.ylabel('Fitness Score')
			plt.xlabel('Iterations ')
			plt.title('Problem: FFP - MIMIC  - Fitness Score vs Iterations ')
			plt.grid()
			plt.legend()
			fig.tight_layout()
			fig.savefig('P2_FFP_MIMIC_fitnessScore.png')

			t1 = time.clock() - t0
			f.write("Runtime %s seconds \n" % (t1))

			print ('MIMIC DONE. Runtime = ', t1)






# *******  PLOTTING  **********



# Plot P1: Continuos Peaks CPP

if (0): # Best
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

	best_rhc = [10.0 ,
				10.0 ,
				9.0 ,
				11.0 ,
				11.0 ,
				10.0 ,
				11.0 ,
				12.0 ,
				15.0 ,
				14.0 ,
				10.0 ,
				10.0 ,
				10.0 ,
				11.0 ,
				11.0 ,
				15.0 ,
				121.0,
				13.0 ,
				13.0 ,
				11.0 ,
				12.0 ,
				13.0 ,
				10.0 ,
				12.0 ,
				12.0 ,
				13.0 ,
				10.0 ,
				12.0 ,
				15.0 ,
				 9.0 ,
				12.0 ,
				17.0 ,
				13.0 ,
				15.0 ,
				17.0 ,
				11.0 ,
				10.0 ,
				16.0 ,
				12.0 ,
				14.0 ,
				9.0 ,
				14.0 ,
				12.0 ,
				12.0 ,
				10.0 ,
				14.0 ,
				15.0 ,
				11.0 ,
				115.0,
				11.0 ]

	best_sa = [141.0,
			   142.0,
			   28.0 ,
			   133.0,
			   29.0 ,
			   50.0 ,
			   136.0,
			   137.0,
			   160.0,
			   143.0,
			   140.0,
			   29.0 ,
			   142.0,
			   147.0,
			   48.0 ,
			   134.0,
			   40.0 ,
			   153.0,
			   140.0,
			   33.0 ,
			   138.0,
			   37.0 ,
			   131.0,
			   31.0 ,
			   141.0,
			   41.0 ,
			   39.0 ,
			   132.0,
			   129.0,
			   28.0 ,
			   157.0,
			   134.0,
			   134.0,
			   138.0,
			   147.0,
			   147.0,
			   52.0 ,
			   38.0 ,
			   131.0,
			   49.0 ,
			   63.0 ,
			   46.0 ,
			   132.0,
			   135.0,
			   142.0,
			   140.0,
			   53.0 ,
			   152.0,
			   37.0 ,
			   153.0]

	best_ga = [126.0,
			   131.0,
			   123.0,
			   129.0,
			   121.0,
			   132.0,
			   125.0,
			   128.0,
			   17.0 ,
			   125.0,
			   125.0,
			   119.0,
			   129.0,
			   121.0,
			   123.0,
			   129.0,
			   126.0,
			   131.0,
			   123.0,
			   125.0,
			   122.0,
			   113.0,
			   127.0,
			   123.0,
			   126.0,
			   121.0,
			   124.0,
			   128.0,
			   126.0,
			   123.0,
			   122.0,
			   123.0,
			   129.0,
			   123.0,
			   124.0,
			   126.0,
			   126.0,
			   124.0,
			   129.0,
			   124.0,
			   126.0,
			   123.0,
			   126.0,
			   129.0,
			   126.0,
			   124.0,
			   123.0,
			   131.0,
			   123.0,
			   122.0]

	x_mimic = [1  ,
	           101,
	           201,
	           301,
	           401,
	           501,
	           601,
	           701,
	           801,
	           901]
	best_mimic = [ 127.0,
				   124.0,
				   127.0,
				   139.0,
				   129.0,
				   125.0,
				   123.0,
				   126.0,
				   137.0,
				   131.0]

	fig = plt.figure(51)

	plt.plot(x ,best_rhc, label="Best RHC Accuracy")
	plt.plot(x ,best_sa, label="Best SA Accuracy")
	plt.plot(x ,best_ga, label="Best GA Accuracy")
	plt.plot(x_mimic ,best_mimic, label="Best MIMIC Accuracy")
	plt.ylabel('Fitness Scores')
	plt.xlabel('Iterations ')
	plt.title('Continuous Peaks Problem (CPP) - Best Fitness Scores vs Iterations ')
	plt.grid()
	plt.legend()
	fig.savefig('P2_CPP_combined.png')

	if (1): # Avg best fitness
		best_rhc = 	  round(np.average(best_rhc),2)
		best_sa = 	  round(np.average(best_sa),2)
		best_ga = 	  round(np.average(best_ga),2)
		best_mimic =  round(np.average(best_mimic),2)
		x =[ 'RHC', 'SA', 'GA', 'MIMIC']
		y = [ best_rhc, best_sa, best_ga, best_mimic]
		print (y)
		fig, ax = plt.subplots()
		ax.bar(x, y, width = 0.5)
		ax.set_ylim([0,150])
		plt.ylabel('Fitness Scores')
		plt.xlabel('Methods')
		ax.set_xticklabels(x)
		for i in range(len(y)):#
			plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')

		ax.yaxis.grid()
		plt.title('P2_CPP_Average Best Fitness Scores Comparison ')
		fig.savefig('P2_CPP_Average Best Fitness Scores.png')


if (0): # Runtime
	a =0
	x =[ 'RHC', 'SA', 'GA', 'MIMIC']
	y = [round(2.1075242999999997/4, 2), round(23.2185963/3,2), round(128.85020640000002/4,2), round(4996.5285638000005*5/6,2)]
	fig, ax = plt.subplots()
	ax.bar(x, y, width = 0.5, color='r')
	plt.ylabel('Runtime (s)')
	plt.xlabel('Methods')
	ax.set_xticklabels(x)
#	ax.set_ylim([0,125])
	for i in range(len(y)):#
		plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')
	ax.yaxis.grid()
	plt.title('P2_CPP_Runtime (seconds) Comparison ')






# Plot P2 - FFP


if (0): # Best
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

	best_rhc = [18.0,
				17.0,
				18.0,
				17.0,
				18.0,
				18.0,
				16.0,
				17.0,
				18.0,
				18.0,
				17.0,
				17.0,
				17.0,
				16.0,
				16.0,
				18.0,
				17.0,
				18.0,
				17.0,
				18.0,
				17.0,
				17.0,
				16.0,
				17.0,
				17.0,
				16.0,
				17.0,
				17.0,
				18.0,
				16.0,
				18.0,
				16.0,
				17.0,
				16.0,
				17.0,
				17.0,
				18.0,
				17.0,
				17.0,
				16.0,
				17.0,
				17.0,
				17.0,
				17.0,
				19.0,
				17.0,
				16.0,
				17.0,
				17.0,
				18.0]

	best_sa = [18.0,
			   18.0,
			   19.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   19.0,
			   18.0,
			   18.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   18.0,
			   18.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   18.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   19.0,
			   18.0,
			   18.0,
			   19.0,
			   18.0]

	best_ga = [17.0,
			   17.0,
			   17.0,
			   17.0,
			   19.0,
			   18.0,
			   17.0,
			   18.0,
			   18.0,
			   18.0,
			   17.0,
			   17.0,
			   16.0,
			   18.0,
			   19.0,
			   17.0,
			   18.0,
			   17.0,
			   17.0,
			   17.0,
			   18.0,
			   19.0,
			   18.0,
			   17.0,
			   19.0,
			   17.0,
			   18.0,
			   18.0,
			   18.0,
			   16.0,
			   17.0,
			   17.0,
			   18.0,
			   17.0,
			   17.0,
			   17.0,
			   17.0,
			   18.0,
			   16.0,
			   17.0,
			   17.0,
			   17.0,
			   17.0,
			   18.0,
			   17.0,
			   18.0,
			   17.0,
			   18.0,
			   18.0,
			   17.0]

	best_mimic = [19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   18.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0,
				   19.0]

	fig = plt.figure(51)

	plt.plot(x ,best_rhc, label="Best RHC Accuracy")
	plt.plot(x ,best_sa, label="Best SA Accuracy")
	plt.plot(x ,best_ga, label="Best GA Accuracy")
	plt.plot(x ,best_mimic, label="Best MIMIC Accuracy")
	plt.ylabel('Fitness Scores')
	plt.xlabel('Iterations ')
	plt.title('Flip Flop Problem (FFP) - Best Fitness Scores vs Iterations ')
	plt.grid()
	plt.legend()
	fig.savefig('P2_FFP_combined.png')

	if (1): # Avg best fitness
		best_rhc = 	  round(np.average(best_rhc),2)
		best_sa = 	  round(np.average(best_sa),2)
		best_ga = 	  round(np.average(best_ga),2)
		best_mimic =  round(np.average(best_mimic),2)
		x =[ 'RHC', 'SA', 'GA', 'MIMIC']
		y = [ best_rhc, best_sa, best_ga, best_mimic]
		print (y)
		fig, ax = plt.subplots()
		ax.bar(x, y, width = 0.5)
		ax.set_ylim([0,20])
		plt.ylabel('Fitness Scores')
		plt.xlabel('Methods')
		ax.set_xticklabels(x)
		for i in range(len(y)):#
			plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')

		ax.yaxis.grid()
		plt.title('P2_FFP_Average Best Fitness Scores Comparison ')
		fig.savefig('P2_FFP_Average Best Fitness Scores.png')


if (0): # Runtime
	a =0
	x =[ 'RHC', 'SA', 'GA', 'MIMIC']
	y = [round(46.1486558/4, 2), round(5.0204336000000005/3,2), round(46.1486558/4,2), round(656.7589188/6,2)]
	fig, ax = plt.subplots()
	ax.bar(x, y, width = 0.5, color='r')
	plt.ylabel('Runtime (s)')
	plt.xlabel('Methods')
	ax.set_xticklabels(x)
	ax.set_ylim([0,125])
	for i in range(len(y)):#
		plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')
	ax.yaxis.grid()
	plt.title('P2_FFP_Runtime (seconds) Comparison ')















# Plot P3: Krapsack KSP

if (1): # Best
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

	best_rhc = [251.0,
				271.0,
				0.0  ,
				253.0,
				0.0 ,
				289.0,
				279.0,
				0.0 ,
				280.0,
				281.0,
				0.0 ,
				275.0,
				281.0,
				241.0,
				0.0 ,
				288.0,
				270.0,
				0.0 ,
				254.0,
				253.0,
				239.0,
				257.0,
				334.0,
				270.0,
				0.0 ,
				279.0,
				0.0 ,
				295.0,
				233.0,
				247.0,
				307.0,
				0.0 ,
				287.0,
				303.0,
				256.0,
				0.0 ,
				305.0,
				286.0,
				272.0,
				317.0,
				287.0,
				0.0 ,
				260.0,
				321.0,
				328.0,
				277.0,
				0.0 ,
				298.0,
				246.0,
				217.0]

	best_sa = [306.0,
			   371.0,
			   334.0,
			   364.0,
			   387.0,
			   322.0,
			   266.0,
			   388.0,
			   378.0,
			   349.0,
			   276.0,
			   358.0,
			   308.0,
			   300.0,
			   271.0,
			   323.0,
			   366.0,
			   380.0,
			   335.0,
			   357.0,
			   345.0,
			   292.0,
			   339.0,
			   406.0,
			   412.0,
			   360.0,
			   366.0,
			   305.0,
			   315.0,
			   397.0,
			   318.0,
			   411.0,
			   391.0,
			   364.0,
			   321.0,
			   366.0,
			   350.0,
			   275.0,
			   309.0,
			   369.0,
			   370.0,
			   360.0,
			   302.0,
			   266.0,
			   319.0,
			   221.0,
			   272.0,
			   308.0,
			   304.0,
			   355.0]

	best_ga = [379.0,
			   376.0,
			   371.0,
			   366.0,
			   366.0,
			   376.0,
			   360.0,
			   372.0,
			   361.0,
			   381.0,
			   368.0,
			   357.0,
			   383.0,
			   382.0,
			   367.0,
			   389.0,
			   401.0,
			   378.0,
			   353.0,
			   364.0,
			   385.0,
			   342.0,
			   373.0,
			   359.0,
			   353.0,
			   407.0,
			   370.0,
			   366.0,
			   351.0,
			   372.0,
			   358.0,
			   370.0,
			   363.0,
			   360.0,
			   381.0,
			   358.0,
			   364.0,
			   383.0,
			   367.0,
			   372.0,
			   384.0,
			   387.0,
			   387.0,
			   392.0,
			   389.0,
			   366.0,
			   394.0,
			   348.0,
			   349.0,
			   365.0]

	best_mimic = [284.0,
			      297.0,
			      272.0,
			      281.0,
			      297.0,
			      299.0,
			      284.0,
			      275.0,
			      262.0,
			      280.0,
			      285.0,
			      274.0,
			      275.0,
			      274.0,
			      255.0,
			      277.0,
			      272.0,
			      289.0,
			      287.0,
			      278.0,
			      273.0,
			      306.0,
			      282.0,
			      282.0,
			      250.0,
			      283.0,
			      269.0,
			      268.0,
			      258.0,
			      292.0,
			      271.0,
			      300.0,
			      265.0,
			      284.0,
			      296.0,
			      275.0,
			      286.0,
			      246.0,
			      286.0,
			      269.0,
			      283.0,
			      308.0,
			      274.0,
			      287.0,
			      319.0,
			      267.0,
			      286.0,
			      277.0,
			      284.0,
			      275.0]

	fig = plt.figure(51)

	plt.plot(x ,best_rhc, label="Best RHC Accuracy")
	plt.plot(x ,best_sa, label="Best SA Accuracy")
	plt.plot(x ,best_ga, label="Best GA Accuracy")
	plt.plot(x ,best_mimic, label="Best MIMIC Accuracy")
	plt.ylabel('Fitness Scores')
	plt.xlabel('Iterations ')
	plt.title('Krapsack Problem (KSP) - Best Fitness Scores vs Iterations ')
	plt.grid()
	plt.legend()
	fig.savefig('P2_KSP_combined.png')

	if (1): # Avg best fitness
		best_rhc = 	  round(np.average(best_rhc),2)
		best_sa = 	  round(np.average(best_sa),2)
		best_ga = 	  round(np.average(best_ga),2)
		best_mimic =  round(np.average(best_mimic),2)
		x =[ 'RHC', 'SA', 'GA', 'MIMIC']
		y = [ best_rhc, best_sa, best_ga, best_mimic]
		print (y)
		fig, ax = plt.subplots()
		ax.bar(x, y, width = 0.5)
	#	ax.set_ylim([0,150])
		plt.ylabel('Fitness Scores')
		plt.xlabel('Methods')
		ax.set_xticklabels(x)
		for i in range(len(y)):#
			plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')

		ax.yaxis.grid()
		plt.title('P2_KSP_Average Best Fitness Scores Comparison ')
		fig.savefig('P2_KPS_Average Best Fitness Scores.png')


if (1): # Runtime
	a =0
	x =[ 'RHC', 'SA', 'GA', 'MIMIC']
	y = [round(1.0062043999999999/4, 2), round(1.7834417/3,2), round(85.6498484/4,2), round(5409.1366506/6,2)]
	fig, ax = plt.subplots()
	ax.bar(x, y, width = 0.5, color='r')
	plt.ylabel('Runtime (s)')
	plt.xlabel('Methods')
	ax.set_xticklabels(x)
#	ax.set_ylim([0,125])
	for i in range(len(y)):#
		plt.annotate(str(y[i]), xy=(x[i],y[i]), ha='center', va='bottom')
	ax.yaxis.grid()
	plt.title('P2_KSP_Runtime (seconds) Comparison ')


plt.show()
