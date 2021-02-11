// Student Name: Orhun Görkem
// Student Number: 2017400171
// Compile Status: Compiling
// Program Status: Working



#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <fstream>
#include <bits/stdc++.h> 
#include <sstream> 

using namespace std;

int instPerProcessor;

//Takes Manhattan Distance of 2 vectors
float ManhattanDistance(vector<float> v1, float* v2){  
    float sum=0;
    for(int i=0;i<v1.size();i++){
        sum+=abs(v1[i]-v2[i]);
    }
    return sum;
}



int main(int argc, char* argv[]){

    int rank; // rank of the current processor
    int size; // total number of processors

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // assigns the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // assigns size to the total number of processors
    
        string line;
        string filename=argv[1];   //input file
        ifstream myfile ("./"+filename);
        int P;  //number of processors
        int N;  //number of instances
        int A;  //number of features
        int M;  //iteration count
        int T;  //number of features to pick for each processor
        
        
    
        myfile>>P;
        myfile>>N;
        myfile>>A;

        int instPerProcessor=N/(P-1);  //each processor share same amout of instances except master
        float instGetter[instPerProcessor][A];    //The 2d float array to keep shared instances
        int classGetter[instPerProcessor];    //Keep classes for shared instances
        float attributes[N+instPerProcessor][A]; //attributes[i]->list of attributes of instance "i-num of instances per processor" 
       
        int classes[N+instPerProcessor]; //classes[i]->class of instance "i-num of instances per processor" 
        int featureGather[A*P];  //resulting features from each processor will be gathered here
        int result[A];   //keeps the resulting features in each processor(before gathering)

        

    if(rank==0){  //master

        //Reading inputs
        myfile>>M;
        myfile>>T;
        float temp;
        int temp2;
            
        for(int i=0;i<N;i++){
            for(int j=0;j<A;j++){
                myfile>>temp;
                attributes[i+instPerProcessor][j]=temp;  //+instPerProcessor since first instPerProcessor indices should be null
                                       //this is because first instPerProcessor  indices are assigned to P0 with scatter and we do not share data to master
            }
            myfile>>temp2;
            classes[i+instPerProcessor]=temp2;   //also read classes
        }
            

        myfile.close();
        
        
    }

    //MPI_SCATTER shares the partitions the data in "attributes" to instGetter in each processor  
    MPI_Scatter(attributes,instPerProcessor*A,MPI_FLOAT,instGetter,instPerProcessor*A,MPI_FLOAT,0,MPI_COMM_WORLD);
    //MPI_SCATTER shares the partitions the data in "classes" to classGetter in each processor 
    MPI_Scatter(classes,instPerProcessor,MPI_INT,classGetter,instPerProcessor,MPI_INT,0,MPI_COMM_WORLD);
    //Broadcast M and T
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&T,1,MPI_INT,0,MPI_COMM_WORLD);
    

    if(rank==0){
        
    }
    
    else{   //slaves

        //instances vector to keep given instances in instGetter
        vector<vector<float>>instances;   //vector was necessary since variable 2d array size can not be declared as function argument
        int numOfFeatures=A; 
        int numOfInstances=instPerProcessor;
    
        for(int i=0;i<instPerProcessor;i++){ //fill instances with copy of instGetter
            vector<float>v;
            for(int j=0;j<numOfFeatures;j++){
                v.push_back(instGetter[i][j]);
            }
            instances.push_back(v);
        }
        
        
        

        float weights[numOfFeatures];  //initiate weights and fill with zeros
        for(int i=0;i<numOfFeatures;i++){
            weights[i]=0.0;
        }

        vector<float>maxdif;
        for(int i=0;i<numOfFeatures;i++){
            float max=instances[0][i];
            float min=instances[0][i];
            
            for(int j=0;j<instances.size();j++){
                float cur=instances[j][i];
                if(cur>max)
                    max=cur;
                if(cur<min)
                    min=cur;
            }
            maxdif.push_back(max-min);
        }


        for(int i=0;i<M;i++){    //iterations
        
            int instClass=classGetter[i];   //get the class
            vector<float>instance=instances[i];  //take the instance in order
            //we will find nearest hit and miss (since diff outputs are always between 0 and 1, -1 indicates the procedure has not started)
            float nearHit=-1.0;      float nearMiss=-1.0;
            float* hitInst;
            float* missInst;
            for(int j=0;j<numOfInstances;j++){
                if(i==j){
                    continue;
                }
                float dist=ManhattanDistance(instance,instGetter[j]);   //get distance of each pair
                bool miss=instClass^classGetter[j];  //miss means classes are different so check with xor
                bool hit=!miss;
                if(miss&&(nearMiss<0)){
                    nearMiss=dist;
                    missInst=instGetter[j];
                    continue;
                }
                if(hit&&(nearHit<0)){
                    nearHit=dist;
                    hitInst=instGetter[j];
                    continue;
                }
                if(hit&&dist<nearHit){
                    nearHit=dist;
                    hitInst=instGetter[j];
                    continue;
                }
                if(miss&&dist<nearMiss){
                    nearMiss=dist;
                    missInst=instGetter[j];
                    continue;
                }
            }

            for(int j=0;j<numOfFeatures;j++){   //update weights with diff
                weights[j]=weights[j]-abs(instance[j]-hitInst[j])/maxdif[j]+ abs(instance[j]-missInst[j])/maxdif[j];
            }
        }


        vector<pair<float, int>>W;   //will keep feature /weight pairs
        pair<float,int>p;
        for(int i=0;i<numOfFeatures;i++){
            p.first=weights[i];
            p.second=i;
            W.push_back(p);
        }
        sort(W.begin(),W.end());    //sort weight featurenum pairs according to weights

        
        for(int i=0;i<T;i++){
            result[i]=W[W.size()-1-i].second;
        }
        if(T>0){
            sort(result,result+T);  //result keeps the important feature numbers sorted
        }
        //Output the sorted sequence
        cout<<"Slave P"<<rank<<" :";
        for(int i=0;i<T;i++){
            cout<<" "<<result[i];
        }
        cout<<"\n";
        
    }

    //MPI_GATHER accumulates the results from each processor
    MPI_Gather(result, T,MPI_INT,featureGather,T,MPI_INT,0,MPI_COMM_WORLD);

    if(rank==0){  //master

        set<int>masterSet;
        for(int i=T;i<T*P;i++){
            masterSet.insert(featureGather[i]);   //use set to automatically remove duplicates
        }   
        vector<int>masterRes;
        copy(masterSet.begin(),masterSet.end(),back_inserter(masterRes));   //copy set to vector

        //Output the result
        cout<<"Master P0 :";
        for(int i=0;i<masterRes.size();i++){
            cout<<" "<<masterRes[i];
        }
    }



    MPI_Finalize();  


    return 0;
}