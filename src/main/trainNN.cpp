#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char* argv[]) {
	ifstream inputFile;
	inputFile.open("data/FEATURES/DR1_FAKS0_SA0001.mfcc");

	for (std::string line; std::getline(inputFile, line); ) {
		std::cout << line << std::endl;
	}
}
