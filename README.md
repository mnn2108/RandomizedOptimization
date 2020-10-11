# RandomizedOptimization

README.txt
Author: MinhTrang (Mindy) Nguyen
Date: Oct 11, 20202


REQUIREMENTS:
    - pandas 1.0.5
    - numpy 1.17.0
    - sklearn 0.23.2
    - matplotlib 3.2.2
    - mlrose 1.3.0



CODE DESCRIPTIONS:
      Source code is in:
      https://github.com/mnn2108/RandomizedOptimization
      The project was divided into 2 parts.
      Part 1 of the project is in main.py, to perform the Neural Network Weight Optimization part of the project. The program will load the input data (PhishingData.csv).
      There are 3 main sections in this part:

            1. Perform Neural Network Weight Optimization using 4 methods:
              - Back Propagation (BP)
              - Random Hill Climb (RHC)
              - Simulated Annealing (SA)
              - Genetic Algorithm (GA)
            For each method above, the program will run through a set of different hyper parameter, plot the performance, and record the value in the text file for later use.

            2. Plotting the best result.
            To plot the best result, I will pick the best performance fitness from each of the technique above and plot it in one chart so that we can have an apple to apple comparison.

            3. Averaging the best fitness and plot the runtime.


      Part 2 of the project is in main2.py, to perform the Analyzing Four Algorithm on the Three Problems.
      There are 3 main sections in this part corresponding to 3 optimization problems. In each problem, there is separate function to run different development of the project.

            1. Problem 1: Continuous Peak Problem (CPP)
                - Calculated fitness score vs Iterations for several different methods:
                      A - Random Hill Climb (RHC)
                      B - Simulated Annealing (SA)
                      C - Genetic Algorithm (GA)
                      D - MIMIC
                - Plotting the best result
                - Average the best fitness of each method and plot the runtime

            2: Problem 2: Flip Flop Problem (FFP)
                - Calculated fitness score vs Iterations for several different methods:
                      A - Random Hill Climb (RHC)
                      B - Simulated Annealing (SA)
                      C - Genetic Algorithm (GA)
                      D - MIMIC
                - Plotting the best result
                - Average the best fitness of each method and plot the runtime

            3: Problem 3: Krapsack Problem (KSP)
              - Calculated fitness score vs Iterations for several different methods:
                    A - Random Hill Climb (RHC)
                    B - Simulated Annealing (SA)
                    C - Genetic Algorithm (GA)
                    D - MIMIC
              - Plotting the best result
              - Average the best fitness of each method and plot the runtime


HOW TO RUN:
To run Part 1 - Neural Network Weight Optimization: python main.py
To run Part 2 - Analyzing Four Algorithm on the Three Problems: python main2.py


DATASETS:
I use the PhishingData.csv for the Part 1 of the project: Neural Network weight analysis.


NOTES:
I only show the source code, input file, and a README in this repository.
All the output data and charts can be regenerate from the codes above.



REFERENCE WEBSITES:
-	https://www.cc.gatech.edu/~isbell/tutorials/mimic-tutorial.pdf
-	https://mlrose.readthedocs.io/
-	https://classroom.udacity.com/

