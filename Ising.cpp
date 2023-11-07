#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <time.h>
#include <ctime>

#include <bits/stdc++.h>
using namespace std;


const int N = 10;
const int counterMax = 1000;
const float J = 1;
const int dimension = 2;
const int totalSpins = pow(N,dimension);
const double criticalTemperature = 2/(log(1+sqrt(2)));
const int rows = 1;
const bool print = false;

const double lowTempCutoff = criticalTemperature;
const double highTempCutoff = criticalTemperature;


std::mt19937 initializeRandomGenerator() {
    unsigned seed = static_cast<unsigned>(std::time(nullptr));
    std::mt19937 rng(seed);
    return rng;
}
void linspace(std::vector<double> &interpVector, float startValue, float endValue, int numPoints) {
    interpVector.clear();
    double step = (endValue - startValue) / (numPoints - 1);

    // Populate the vector with equally spaced values
    for (int n = 0; n < numPoints; n++) {
        interpVector.push_back(startValue + step * n);
    }
}
int getElement(vector<int> matrix, int address) {
    return matrix[address];
}
void setElement(vector<int> &matrix, int address, int val) {
    matrix[address] = val;
}
void initializeLattice(vector<int> &lattice, std::mt19937 &rng) {
    std::uniform_int_distribution<int> coin(0, 1);

    for (int i = 0; i < lattice.size(); i++) {
        int spin = (coin(rng) == 0) ? -1 : 1;
        lattice[i] = spin;
    }
}
vector<int> getNeighbours(int site) {
    vector<int> neighbours ={};

    int posSite, negSite;

    for (int d = 1;d <= dimension; d++) {
        
        posSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site + int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );
        negSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site - int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );      
        
        neighbours.push_back(posSite);
        neighbours.push_back(negSite);
        
    }
    return neighbours;
}
void showLattice(vector<int> &lattice, bool showValues = false ) {
    
    for (int site = 0; site < totalSpins; site++) {
        
        for (int dim = dimension; dim > 0; dim--){
            if (site %(int(pow(N,dim))) == 0) {
                cout << "\n";
            }            
        } 

        if (showValues){
            
            if (lattice[site] < 0){
                cout << lattice[site] << ",";
            }
            else{
                cout << " " << lattice[site] << ",";
            }
        }
        else{
            if (lattice[site] == 1) {
                cout << "+ ";
            } 
            else {
                cout << "- ";
            }
        }
    }
    
    


    int plusCount = 0, minusCount = 0;
    int agrees = 0, disagrees = 0;

    double magnetisation = 0;
    for (int site = 0; site < totalSpins; site++) {
        
        magnetisation = magnetisation + getElement(lattice,site);
        vector<int> siteNeighbours = getNeighbours(site);

        for (int dir = 0; dir < siteNeighbours.size(); dir++) {
            if (lattice[site]*lattice[siteNeighbours[dir]] ==1) {
                agrees++;
            } 
            else{
                disagrees++;
            }
        } 
    }

    
   
    

    std::cout << endl;

    std::cout << "Plus count: " << plusCount << " " <<"Minus count: " << minusCount <<"\n";
    std::cout << "Agrees: " << agrees << " " <<"Disagrees: " << disagrees <<"\n";
    std::cout << "Magnetisation: " << magnetisation << " " <<"Mean Magnetisation: " << magnetisation / (totalSpins) <<"\n";
    
}
void flipSpins(vector<int> &lattice, int site) {
        lattice[site] = -1*lattice[site];
}
void flipSpins(vector<int> &lattice, vector<int> sites) {
    for (int i = 0; i < sites.size(); i++) {
        lattice[sites[i]] = -1*lattice[sites[i]];  // flip the spin
    }
}
vector<int> buildCluster(vector<int> &lattice, int startSite, float temperature, std::mt19937&rng ) {

    int startState = getElement(lattice,startSite);

    vector<int> cluster = {startSite};
    vector<int> stackOld = {startSite};



    while (not stackOld.empty()) {
            vector<int> stackNew = {};
            
            // For all members of stack Old
            for (int i=0;i<stackOld.size();i++) {
                
                // Get the neighbours
                vector<int> neighbs = getNeighbours(stackOld[i]);
                
                //For each neighbour
                for (int j=0;j<neighbs.size();j++) {
                    
                    //if it isn't in the cluster
                    if ( find(cluster.begin(), cluster.end(), neighbs[j]) != cluster.end() ) {
                    }
                    else {
                        // If same state as start
                        if (getElement(lattice, neighbs[j]) == startState) {
                            uniform_real_distribution<double> distribution(0.0, 1.0);
                            if  (distribution(rng) < (1 - exp(-2 * J / temperature))){
                                stackNew.push_back(neighbs[j]);
                                cluster.push_back(neighbs[j]);
                            }
                        }
                    }
                }
                stackOld = stackNew;
            }

        }
    return cluster;
}
vector<double> getTemperatures(double lowTempCutoff,double highTempCutoff){

    vector<double> temperatures(rows);
    linspace(temperatures, lowTempCutoff, highTempCutoff,rows);    
    return temperatures;
}
void write_out(string fileName, vector<int>& vect) {
    ofstream outFile(fileName, ios::out | ios::binary);

    int size = sizeof(vect[0]);
    std::cout << "size is " << size << endl;

    for (int i = 0; i < vect.size(); i++) {
        outFile.write(reinterpret_cast<char*>(&vect[i]), size);
    }
}
void write_out(const std::string& fileName, const std::vector<std::vector<int>>& vect) {
    std::ofstream outFile(fileName, std::ios::out | std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (const auto& row : vect) {
        for (const int element : row) {
            outFile.write(reinterpret_cast<const char*>(&element), sizeof(element));
        }
    }

    outFile.close();
}
int main()
{
    
    std::mt19937 rng = initializeRandomGenerator();
    vector<double> temperatures = getTemperatures(lowTempCutoff, highTempCutoff);
    vector<int> lattice(totalSpins,0);
    vector<int> labels(rows);

    // Set up output vector:
    vector<vector<int>> output(rows,vector<int>(totalSpins,0));
    double overallSum = 0;
    double count = 0;
    int firstSite = 0;
    int secondSite = 0;

    for (int t = 0; t < rows; t++){
        
        uniform_int_distribution<int> distrib(0, totalSpins-1);
        float temperature = temperatures[t];

        if (temperature > criticalTemperature){
            labels[t] = 1;
        } else{
            labels[t] = 0;
        }

        initializeLattice(lattice,rng);
        double initialSum = 0;
        double finalSum = 0;

        for (int i=0;i<totalSpins;i++){
            initialSum += lattice[i];
        }

        


        int startSite = distrib(rng);
        firstSite = startSite;
        int startState;

        
        for (int counter = 0; counter < counterMax; counter++) {
            startSite = distrib(rng);
            if (counter == 1){
                secondSite = startSite;
            }
            vector<int> cluster = buildCluster(lattice, startSite, temperature, rng);
            




            flipSpins(lattice, cluster);
        }


        
        for (int site = 0; site < totalSpins; site++){
            output[t][site] = int((lattice[site]+1)/2);
            finalSum += lattice[site];
            overallSum += lattice[site];
            count ++;
            if (site%10 == 0 ){
                cout << endl;
            }
            cout << output[t][site] << ", ";

        }
        cout << endl;
        //cout << sum/count  << endl;
        


       double initialMean = initialSum/totalSpins;
       double finalMean = finalSum/totalSpins;

       if (t%1 == 0) {
            std::cout << "Test " << t+1 << std::setprecision(15)<< ", temperature: " << temperature << ", Inital mean is: " << std::setprecision(15)<< initialMean  << ", final mean: " << std::setprecision(15)<< finalMean<< ", first site: " << firstSite << ", second site: " << secondSite <<endl;
       }
       

    }

    if (print){
        write_out("C:\\Users\\jjhadley\\Documents\\Projects\\Ising\\Data\\testingData.dat",output);
        write_out("C:\\Users\\jjhadley\\Documents\\Projects\\Ising\\Data\\testingLabels.dat",labels);
        cout << "Written" << endl;
    }
    

    std::cout << overallSum/count  << endl;
    std::cout << "Done!";

    return 0;


}
