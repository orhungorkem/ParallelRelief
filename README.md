# Parallel Relief

In supervised machine learning, we have labeled input data which consist of feature values and
class label for every instance. Class label is a single target field whereas there may be thousands
of features. The aim of machine learning is to predict the class label by learning a pattern of
features. Irrelevant features may decrease the accuracy of your decision models. That's why, feature selection is a must.
Relief algorithm works for eliminating the unnecessary features for a given class.
In this Project, our aim was to implement a relief algorithm with the performance increasing effect of MPI (Message Passing Interface). 
The program execution is conducted by of one master and several slave processors. Since there can be high number
of instances to examine for feature selection, the job of examination should be shared among some additional processors. 
Since single instruction is executed on multiple data, we can say that SIMD (Single Instruction Multiple Data) machine is used for this program.

## Requirements and Execution

The code is written with Open-MPI 4.0.5. Therefore, installation of Open-MPI should be made before execution.
For really big inputs, to avoid stack overflow, the following command is recommended:
> ulimit -S -s 131072
<!-- end of the list -->
Then, directory with the mpi_relief.cpp file and test cases including the instances should be navigated. 
After that, the commands below will be sufficient:
* mpic++ -o mpi_relief ./mpi_relief.cpp
* mpirun --oversubscribe -np <num_of_processors> mpi_relief <input_file>
<!-- end of the list -->
where input file is the relative path for input file. \

## Input/Output
Input file should include:
* Number of processors to use : P
* Number of instances to apply relief : N
* Number of features in each instance: A
* Number of iterations in relief algorithm : M
* Number of features to select for each processor : T
* N lines of input that keeps all features and classes of each instance

Example:
```
6
50	10	5	2
35.7581	71.3776	21.7663	6.9562	72.687	50.5385	245.6104	1.942	11.1582	15.5402	0
2.1202	144.0322	12.5488	4.2346	11.9353	4.1862	436.0972	2.5426	1.0975	0.0941	1
8.8889	78.1114	9.787	2.7225	28.1304	30.9039	502.0733	2.2641	12.5105	0.135	0
4.1047	39.976	4.8287	0.4912	11.2183	3.2917	126.2577	0.2	113.5855	0.1575	1
7.0183	125.647	23.0461	3.4679	29.3523	13.8164	967.8378	2.1142	23.4257	0.0182	0
1.1313	27.6305	3.5262	0.0952	3.2275	0.4476	163.3341	0.8204	213.9672	0.0043	1
14.7859	145.5234	11.285	6.6207	56.6405	32.9578	470.9935	3.1088	10.6038	0.0716	0
1.3874	34.4677	9.5864	0.3819	4.3961	5.766	121.3457	1.2861	103.3551	0.0407	1
6.1965	176.8086	33.1594	2.6696	78.7756	25.9433	941.9163	1.4192	13.4956	0.0748	0
0.4791	16.3363	7.2485	0.126	7.3098	0.2786	278.4067	0.1402	119.4868	0.0101	1
12.4904	54.979	22.8783	1.7211	25.1298	22.4119	648.8583	1.528	4.7716	0.0054	0
1.5832	135.245	4.265	3.1558	17.2276	12.5894	465.5874	1.5759	2.8262	0.1774	1
9.7198	128.8188	40.1136	5.5082	71.5397	59.0523	767.3074	1.5329	8.0167	0.0386	0
2.127	43.3099	4.9138	0.6773	17.5914	3.9391	171.1459	0.2523	67.35	0.0264	1
14.8682	186.3404	31.5342	2.6863	30.2744	33.0476	598.1636	0.8386	29.6513	0.0181	0
1.4263	48.5682	2.5638	1.4722	18.1361	9.0385	254.3237	1.1722	115.8405	0.0442	1
8.4257	158.5447	12.2555	3.9583	102.3578	36.6986	597.572	1.7221	10.7283	0.0413	0
2.1961	46.1035	3.0222	0.8626	10.8032	1.5389	249.6151	0.4137	49.4484	0.0167	1
3.3346	56.1982	6.2937	1.7711	124.9743	20.8603	296.035	1.5341	7.8409	2.1397	0
3.3641	88.4442	4.9784	1.1991	14.5201	6.4742	153.6003	0.2522	63.8406	0.0	1
17.5627	156.5368	55.6233	3.3004	97.0171	30.0423	907.4357	1.0965	33.1623	0.1395	0
1.0823	39.0491	3.6542	0.652	9.649	1.7984	155.1681	0.7737	103.7561	0.0196	1
2.1042	34.6533	9.2914	3.1866	24.1667	20.0258	673.4403	2.3934	52.1626	6.6743	0
1.3707	55.6768	20.1208	0.4543	6.2899	2.4976	115.2409	0.3765	71.5964	0.08	1
4.011	89.5595	7.6762	6.5587	31.0014	33.3964	135.2445	5.069	27.4413	0.1397	0
0.8167	87.161	0.2356	1.6832	3.4837	4.1885	174.6963	1.2638	14.7665	0.244	1
11.0264	109.3926	28.198	4.4082	70.7409	22.8031	585.207	1.9488	15.129	0.0442	0
2.1691	70.8011	4.3842	0.6751	7.5961	2.863	137.8216	0.4692	122.8586	0.0042	1
5.3799	114.2304	6.5871	4.3647	16.985	43.4546	363.483	5.3367	10.4892	0.0122	0
1.7932	55.3736	2.0922	1.6159	5.9457	11.2728	112.4495	0.6008	108.1532	0.0373	1
10.3249	114.8819	6.2965	4.4255	42.7977	30.7644	202.454	2.3506	29.7831	1.3941	0
2.0763	69.323	3.153	1.3401	3.1837	1.822	95.121	0.4327	58.3096	0.0057	1
12.8518	192.1551	8.9723	3.919	166.6205	18.1273	203.1516	1.5322	5.9807	0.0058	0
1.4245	46.3991	2.0969	0.6553	5.8534	1.3309	160.0858	0.6011	130.8683	0.019	1
7.2317	140.5153	5.8442	4.412	115.9865	15.0132	120.9083	1.8049	21.757	0.2741	0
2.4079	28.2529	5.7466	0.7056	3.8996	3.0249	148.1324	0.477	33.6508	0.028	1
10.8072	101.4663	15.4841	6.0324	68.9887	19.8363	290.5995	2.5993	12.2391	0.2412	0
0.9371	41.8566	3.4674	0.3126	8.2485	2.1805	197.8539	1.1183	112.6788	0.0114	1
3.4988	40.3987	7.986	1.6129	20.5285	27.2328	347.8643	0.8441	6.7911	0.0639	0
1.5662	49.6358	4.9762	0.3496	6.3473	3.0422	143.4884	0.841	135.6122	0.0	1
15.6584	63.9392	37.0633	2.2212	95.9134	52.369	447.74	1.1908	15.5339	4.7097	0
1.2637	43.0234	3.1199	0.5363	13.9656	3.3893	126.3183	1.2149	134.1554	0.0163	1
10.1149	135.491	20.4541	7.1536	24.5801	20.8528	596.964	4.8943	29.0273	0.1123	0
2.0357	60.0575	3.6502	0.8871	3.4251	1.1229	122.8195	0.2856	97.2178	0.0035	1
9.2441	181.1531	4.1992	7.1134	109.0599	93.4784	307.0906	6.8466	2.932	0.0779	0
1.5704	64.2543	11.5959	1.0698	30.2961	1.1014	317.2043	1.3171	41.7679	0.0041	1
12.684	167.4428	3.6246	6.2451	14.4537	46.7977	428.4485	7.7573	19.077	0.0037	0
0.888	52.8229	1.626	0.6006	12.2125	3.1418	154.4308	1.0818	102.1502	0.0333	1
14.2178	188.6772	18.17	2.3081	75.985	42.5701	419.9249	0.6874	5.4592	0.0263	0
0.8235	24.752	2.3301	0.2578	12.6706	1.6863	182.2044	1.2055	108.6792	0.0045	1

```
The output is printed to terminal and indicates the top features selected by each slave processor. Master processor unions the results from slaves.
Example:
```
Slave P1 : 0 5
Slave P2 : 0 2
Slave P3 : 5 7
Slave P4 : 5 7
Slave P5 : 0 3
Master P0 : 0 2 3 5 7

```
## Implementation of Parallelism

Parallel programs using MPI should be initiated with the following code segment.
```
    int rank; // rank of the current processor
    int size; // total number of processors

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // assigns the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // assigns size to the total number of processors
```

Then program begins with master processor. Master starts to read input file and stores the instances, feature values, number of slaves, number of features to select etc.
Obeying the working principle of SIMD machine, master processor distributes the input data to the slaves equally using `MPI_Scatter()`. `MPI_Scatter()` partitions a list of data and sends them to corresponding processors. Iteration count and number of features to select should also be passed to slaves, bu it is done with `MPI_Bcast` since such data should not be partitioned, it should just be distributed to all other processors.

```
    //MPI_SCATTER shares the partitions the data in "attributes" to instGetter in each processor  
    MPI_Scatter(attributes,instPerProcessor*A,MPI_FLOAT,instGetter,instPerProcessor*A,MPI_FLOAT,0,MPI_COMM_WORLD);
    //MPI_SCATTER shares the partitions the data in "classes" to classGetter in each processor 
    MPI_Scatter(classes,instPerProcessor,MPI_INT,classGetter,instPerProcessor,MPI_INT,0,MPI_COMM_WORLD);
    //Broadcast M and T
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&T,1,MPI_INT,0,MPI_COMM_WORLD);
```

After data distribution, slaves and master execute different parts of the code. Each processor has an assigned rank, and rank for master is 0. Hence, `if(rank==0)` is a good indicator for master and slave seperation. Slaves apply the relief algorithm with their own data in parallel. However, the relief algorithm is not our main focus here. When slaves complete their job, master collects the results with 

```
    //MPI_GATHER accumulates the results from each processor
    MPI_Gather(result, T,MPI_INT,featureGather,T,MPI_INT,0,MPI_COMM_WORLD);
```
Selected features are available for the master now.



