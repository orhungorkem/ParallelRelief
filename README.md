# Parallel Relief

In supervised machine learning, we have labeled input data which consist of feature values and
class label for every instance. Class label is a single target field whereas there may be thousands
of features. The aim of machine learning is to predict the class label by learning a pattern of
features. Irrelevant features may decrease the accuracy of your decision models. That's why, feature selection is a must.
Relief algorithm works for eliminating the unnecessary features for a given class.
In this Project, our aim was to implement a relief algorithm with the performance increasing effect of MPI (Message Passing Interface). 
The program execution is conducted by of one master and several slave processors. Since there can be high number
of instances to examine for feature selection, the job of examination should be shared among some additional processors. 
Since single instruction is executed on multiple data, we can say that SIMD machine is used for this program.

## Requirements and Execution

The code is written with Open-MPI 4.0.5. Therefore, installation of Open-MPI should be made before execution.
For really big inputs, to avoid stack overflow, the following command is recommended:
> ulimit -S -s 131072
<!-- end of the list -->
Then, directory with the mpi_relief.cpp file and test cases including the instances should be navigated. 
After that, the commands below will be sufficient:
* mpic++ -o mpi_relief ./mpi_relief.cpp
* mpirun --oversubscribe -np <num_of_processors> mpi_relief <inputfile>
<!-- end of the list -->
where input file is the relative path for input file. \
Input file should include:
* Number of processors to use : P
* Number of instances to apply relief : N
* Number of features in each instance: A
* Number of iterations in relief algorithm : M
* Number of features to select for each processor : T
* N lines of input that keeps all features and classes of each instance
