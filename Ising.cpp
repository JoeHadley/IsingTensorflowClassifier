#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <time.h>
#include <ctime>


#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

auto start = high_resolution_clock::now();

const int N = 10;
const int counterMax = 10*N*N;
const float J = 1;
const int dimension = 2;
const int totalSpins = pow(N,dimension);
const double criticalTemperature = 2/(log(1+sqrt(2)));

const int temps = 100;
const int samplesPerTemp = 100;

const int rows = temps*samplesPerTemp;

const double lowTempCutoff = 1;
const double highTempCutoff = 1.5;

const bool writeOut= false;
const bool printSites = false;
const bool printMeans = true;



const string path = "C:\\Users\\jjhadley\\Documents\\Projects\\Ising\\Data\\";
const string folder = "L=" + to_string(N) + "\\";
//const string T_Equals = "T=";
const string mode = "validating";
mt19937 initializeRandomGenerator() {
    unsigned seed = static_cast<unsigned>(time(nullptr));
    mt19937 rng(seed);
    return rng;
}

template <typename T>
void show(const vector<T>& vec, string string = "") {
    
    std::cout << string;
    for (const T& element : vec) {
        std::cout << element << ", ";
    }
    std::cout << endl;
}
void linspace(vector<double> &interpVector, double startValue, double endValue, int numPoints) {
    interpVector.clear();

    if (numPoints == 1){
        interpVector.push_back(startValue);
    }
    else {
        double step = (endValue - startValue) / (numPoints - 1);
        for (int n = 0; n < numPoints; n++) {
            interpVector.push_back(startValue + step * n);
        }
    }
}

/*
vector<double> getTemperatures(double lowTempCutoff,double highTempCutoff){

    vector<double> temperatures(temps);
    linspace(temperatures, lowTempCutoff, highTempCutoff,temps);
    
    vector<double> returnMatrix(rows);
    
    for (int t = 0; t < temps; t++){
        
        for (int s = 0; s < samplesPerTemp; s++){
            returnMatrix[t*samplesPerTemp + s] = temperatures[t];
        }
    }
    return returnMatrix;
}
*/




int getElement(vector<int> matrix, int address) {
    return matrix[address];
}
void setElement(vector<int> &matrix, int address, int val) {
    matrix[address] = val;
}
void initializeLattice(vector<int> &lattice, mt19937 &rng) {
    uniform_int_distribution<int> coin(0, 1);

    for (int i = 0; i < lattice.size(); i++) {
        int spin = (coin(rng) == 0) ? -1 : 1;
        lattice[i] = spin;
    }
}


void getNeighbours(int *neighbours, int size, int site) {

    int neighbSite;

    for (int i = 0;i < size; i++) {
        int d = ceil(0.5*(i+1));

        neighbSite = int(pow(N, d)) * (site / int(pow(N, d))) + (site + int(pow(-1, i)) * int(pow(N, d - 1)) + int(pow(N, dimension))) % int(pow(N, d));

        neighbours[i] = neighbSite;
    }
    
}



void getNeighbours(vector<int> &neighbours, int site) {

    int posSite, negSite;

    for (int d = 1;d <= dimension; d++) {
        
        posSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site + int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );
        negSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site - int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );      
        
        neighbours.push_back(posSite);
        neighbours.push_back(negSite);
        
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
                std::cout << "\n";
            }            
        } 

        if (showValues){
            
            if (lattice[site] < 0){
                std::cout << lattice[site] << ",";
            }
            else{
                std::cout << " " << lattice[site] << ",";
            }
        }
        else{
            if (lattice[site] == 1) {
                std::cout << "+ ";
            } 
            else {
                std::cout << "- ";
            }
        }
    }
    
    


    int plusCount = 0, minusCount = 0;
    int agrees = 0, disagrees = 0;

    double magnetisation = 0;
    for (int site = 0; site < totalSpins; site++) {
        
        magnetisation = magnetisation + getElement(lattice,site);

        vector<int> siteNeighbours(2*dimension);
        getNeighbours(siteNeighbours, site);
        //vector<int> siteNeighbours = getNeighbours(site);


        if (lattice[site] == 1){
            plusCount++;
        }
        else {
            minusCount++;
        }



        for (int dir = 0; dir < 2*dimension; dir++) {
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
vector<int> buildCluster(vector<int> &lattice, int startSite, float temperature, mt19937&rng ) {

    int startState = getElement(lattice,startSite);

    vector<int> cluster = {startSite};
    vector<int> stackOld = {startSite};



    while (not stackOld.empty()) {
            vector<int> stackNew = {};
            
            // For all members of stack Old
            for (int i=0;i<stackOld.size();i++) {
                
                // Get the neighbours

                vector<int> neighbs(2*dimension);
                getNeighbours(neighbs, stackOld[i]);


                //For each neighbour
                for (int j=0;j<2*dimension;j++) {
                    
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

template <typename T>
void write_out(const string& fileName, const vector<T>& vect) {
    ofstream outFile(fileName, ios::out | ios::binary);

    if (!outFile.is_open()) {
        cerr << "Failed to open the file for writing." << endl;
        return;
    }

    for (const T& element : vect) {
        outFile.write(reinterpret_cast<const char*>(&element), sizeof(vect[0]));
    }

    outFile.close();
}
template <typename T>
void write_out(const string& fileName, const vector<vector<T>>& vect) {
    ofstream outFile(fileName, ios::out | ios::binary);

    if (!outFile.is_open()) {
        cerr << "Failed to open the file for writing." << endl;
        return;
    }

    for (const auto& row : vect) {
        for (const T& element : row) {
            outFile.write(reinterpret_cast<const char*>(&element), sizeof(element));
        }
    }

    outFile.close();
}


int main()
{
    mt19937 rng = initializeRandomGenerator();
    //vector<double> temperatures = getTemperatures(lowTempCutoff, highTempCutoff);
    vector<double> temperatures(temps);
    linspace(temperatures,lowTempCutoff,highTempCutoff,temps);
    
    vector<int> lattice(totalSpins,0);

    // Set up output vector:
    vector<double> outputParams = {dimension,N,temps,samplesPerTemp,lowTempCutoff,highTempCutoff};
    vector<string> outputParamDescription = {"dimension","sidelength","temperatureNumber","sampleNumber","lowTempCutoff","highTempCutoff"};



    vector<vector<int>> outputLattices(rows,vector<int>(totalSpins,0));
    vector<int> outputLabels(rows);
    vector<double> outputTemperatures(rows);
    vector<int> outputTNumbers(rows);


    double overallSum = 0;
    double count = 0;
    int firstSite = 0;
    int secondSite = 0;

    float temperature;
    double initialSum;
    double finalSum;
    double initialMean;
    double finalMean;
    int startSite;
    int startState;
    int row = 0;


    for (int t = 0; t < temps; t++){

        for (int s = 0; s < samplesPerTemp; s++){
            
            row = t*samplesPerTemp + s;

            uniform_int_distribution<int> distrib(0, totalSpins-1);
            temperature = temperatures[t];







            initializeLattice(lattice,rng);
            initialSum = 0;
            finalSum = 0;

            for (int i=0;i<totalSpins;i++){
                initialSum += lattice[i];
            }

            


            startSite = distrib(rng);
            firstSite = startSite;
            

            
            for (int counter = 0; counter < counterMax; counter++) {
                startSite = distrib(rng);
                if (counter == 1){
                    secondSite = startSite;
                }
                vector<int> cluster = buildCluster(lattice, startSite, temperature, rng);
                




                flipSpins(lattice, cluster);
            }


            


            // Populate output data
            for (int site = 0; site < totalSpins; site++){
                outputLattices[row][site] = int((lattice[site]+1)/2);
                finalSum += lattice[site];
                overallSum += lattice[site];
                count ++;
                if (site%10 == 0 ){
                    //std::cout << endl;
                }
                //std::cout << output[t][site] << ", ";
            }

            
            if (temperature > criticalTemperature){
                outputLabels[row] = 1;
            } else{
                outputLabels[row] = 0;
            }
            outputTNumbers[row] = t;
            outputTemperatures[row] = temperature;

            


            initialMean = initialSum/totalSpins;
            finalMean = finalSum/totalSpins;

            if (s%1 == 0) {

                    if (samplesPerTemp > 1) {
                        std::cout << "Trial " << row+1 << ", Temp " << t+1<< "/" << temps << ", Sample " << (s)%(samplesPerTemp)+1 <<"/" << samplesPerTemp << ", temperature: " << temperature;
                    }
                    else {
                        std::cout << "Trial " << row+1 << "/" << rows << ", Temperature: " << temperature;
                    }

                    if (printMeans){
                        std::cout << ", Inital mean is: " << setprecision(15)<< initialMean  << ", final mean: " << setprecision(15)<< finalMean;
                    }

                    
                    if (printSites){
                        std::cout << ", first site: " << firstSite << ", second site: " << secondSite;
                    }
                    std::cout << endl;
            }
        

        }

    }

        if (writeOut){
            //string tempString = to_string(t) + "of" + to_string(temps);
        

            write_out(path + folder  + mode +  "Data.dat",outputLattices);
            write_out(path + folder  + mode +  "Labels.dat",outputLabels);
            write_out(path + folder  + mode +  "TNumbers.dat",outputTNumbers);
            write_out(path + folder  + mode +  "Temps.dat",outputTemperatures);
            write_out(path + "params.dat",outputParams);
            write_out(path + "paramsDescription.dat",outputParamDescription);
            //write_out(path + folder  + mode +  "Temps.dat",temperatures);
            //std::cout << "Written to "<< path + folder + tempString + mode +  "Data.dat" << endl;
            //std::cout << "Written to "<< path+folder+mode << endl;
        }
    

    showLattice(lattice);

    std::cout << "Done!"<< endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() / 1e6 << " seconds" << endl;


    return 0;



}
