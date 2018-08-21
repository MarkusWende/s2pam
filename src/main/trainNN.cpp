#include "textgrid.h"

using namespace std;

int main(int argc, char* argv[]) {
	ifstream inputMfccFile;
	inputMfccFile.open("data/FEATURES/DR1_FAKS0_SA0001.mfcc");
	string textGridFilename = "data/TIMIT/TextGrids/DR1_FAKS0_SA0001.TextGrid";

	Textgrid tg(textGridFilename.c_str());

	int i = 0;
	for (string line; getline(inputMfccFile, line); ) {
		//std::cout << line << std::endl;
		i++;
	}

	//tg.print_textgrid_struct();
}
