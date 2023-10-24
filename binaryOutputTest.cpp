#include<iostream>
#include<fstream>
using namespace std;


int main() {
    // Open a binary file for writing
    std::ofstream outFile("data.dat", std::ios::out | std::ios::binary);

    if (!outFile) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 1;
    }

    // Data to be written (for example, an integer and a double)
    int intValue = 42;
    double doubleValue = 3.14159;

    cout << "Fine";

    // Write the data to the file
    outFile.write(reinterpret_cast<char*>(&intValue), sizeof(intValue));
    outFile.write(reinterpret_cast<char*>(&doubleValue), sizeof(doubleValue));

    // Close the file
    outFile.close();

    return 0;
}
