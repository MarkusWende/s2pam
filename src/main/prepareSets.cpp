#include <experimental/filesystem>
#include <string.h>
#include <essentia/algorithmfactory.h>

#include "helper.h"
#include "render.h"
#include "textgrid.h"

using namespace std;
using namespace essentia;
namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[])
{
	string mfccDir = "./data/FEATURES";
	string textGridDir = "./data/TIMIT/TextGrids";

	fs::directory_iterator mfccDirIter(mfccDir), eMfcc;
	std::vector<fs::path> mfccFileList(mfccDirIter, eMfcc);
	fs::directory_iterator textGridDirIter(textGridDir), eTextGrid;
	std::vector<fs::path> textGridFileList(textGridDirIter, eTextGrid);

	if (mfccFileList.size() != textGridFileList.size())
	{
		E_ERROR("\tMFCC File List size differs from TextGrid File List size\n");
		exit(1);
	}

	string mfccFilename;
	string textGridFilename;
	string filename;
	bool finished = false;
	int j = 0;
	int countSI = 0;
	int countSA = 0;
	int countSX = 0;

	/// Main Loop over all mfcc and textGrid files in the corresponding folders
	while(!finished)
	{
		filename = mfccFileList[j];
		mfccFilename = filename;
		
		size_t found = filename.find_last_of("_");
		string sentenceType = filename.substr(found+1);
		found = sentenceType.find_last_of(".");
		sentenceType = sentenceType.substr(0,found);
		sentenceType = sentenceType.substr(0,2);

		if (!sentenceType.compare("SI"))
			countSI++;

		if (!sentenceType.compare("SX"))
			countSX++;

		if (!sentenceType.compare("SA"))
			countSA++;

		j++;
		if(j >= mfccFileList.size())
			finished = true;
	}

	finished = false;
	j = 0;
	int coreTestCount = 0;
	int devTestCount = 0;
	int trainCount = 0;

	/// Main Loop over all mfcc and textGrid files in the corresponding folders
	while(!finished)
	{
		filename = mfccFileList[j];
		mfccFilename = filename;

		size_t found = filename.find_last_of("/\\");
		textGridFilename = filename.substr(found+1);
		found = textGridFilename.find_last_of(".");
		textGridFilename = textGridFilename.substr(0,found);
		textGridFilename.append(".TextGrid");
		textGridFilename.insert(0,"./data/TIMIT/TextGrids/");
		
		found = filename.find_last_of("_");
		string sentenceType = filename.substr(found+1);
		found = sentenceType.find_last_of(".");
		sentenceType = sentenceType.substr(0,found);
		sentenceType = sentenceType.substr(0,2);

		if (sentenceType.compare("SA"))
		{
			//cout << "MFCC: " << mfccFilename << "\tTextGrid: " << textGridFilename << endl;
			//cout << "(" << j+1 << "|" << mfccFileList.size() << ")" << endl;
			int setType = 0;

			found = filename.find_first_of("_");
			string speaker = filename.substr(found+1);
			found = speaker.find_last_of("_");
			speaker = speaker.substr(0,found);

			/// Dev test set						/// Speaker
			if (!speaker.compare("FAKS0")			/// 1
					|| !speaker.compare("MMDB1")	/// 2
					|| !speaker.compare("MBDG0")	/// 3
					|| !speaker.compare("FEDW0")	/// 4
					|| !speaker.compare("MTDT0")	/// 5
					|| !speaker.compare("FSEM0")	/// 6
					|| !speaker.compare("MDVC0")	/// 7
					|| !speaker.compare("MRJM4")	/// 8
					|| !speaker.compare("MJSW0")	/// 9
					|| !speaker.compare("MTEB0")	/// 10
					|| !speaker.compare("FDAC1")	/// 11
					|| !speaker.compare("MMDM2")	/// 12
					|| !speaker.compare("MBWM0")	/// 13
					|| !speaker.compare("MGJF0")	/// 14
					|| !speaker.compare("MTHC0")	/// 15
					|| !speaker.compare("MBNS0")	/// 16
					|| !speaker.compare("MERS0")	/// 17
					|| !speaker.compare("FCAL1")	/// 18
					|| !speaker.compare("MREB0")	/// 19
					|| !speaker.compare("MJFC0")	/// 20
					|| !speaker.compare("FJEM0")	/// 21
					|| !speaker.compare("MPDF0")	/// 22
					|| !speaker.compare("MCSH0")	/// 23
					|| !speaker.compare("MGLB0")	/// 24
					|| !speaker.compare("MWJG0")	/// 25
					|| !speaker.compare("MMJR0")	/// 26
					|| !speaker.compare("FMAH0")	/// 27
					|| !speaker.compare("MMWH0")	/// 28
					|| !speaker.compare("FGJD0")	/// 29
					|| !speaker.compare("MRJR0")	/// 30
					|| !speaker.compare("MGWT0")	/// 31
					|| !speaker.compare("FCMH0")	/// 32
					|| !speaker.compare("FADG0")	/// 33
					|| !speaker.compare("MRTK0")	/// 34
					|| !speaker.compare("FNMR0")	/// 35
					|| !speaker.compare("MDLS0")	/// 36
					|| !speaker.compare("FDRW0")	/// 37
					|| !speaker.compare("FJSJ0")	/// 38
					|| !speaker.compare("FJMG0")	/// 39
					|| !speaker.compare("FMML0")	/// 40
					|| !speaker.compare("MJAR0")	/// 41
					|| !speaker.compare("FKMS0")	/// 42
					|| !speaker.compare("FDMS0")	/// 43
					|| !speaker.compare("MTAA0")	/// 44
					|| !speaker.compare("FREW0")	/// 45
					|| !speaker.compare("MDLF0")	/// 46
					|| !speaker.compare("MRCS0")	/// 47
					|| !speaker.compare("MAJC0")	/// 48
					|| !speaker.compare("MROA0")	/// 49
					|| !speaker.compare("MRWS1")	/// 50
					)
			{
				devTestCount++;
				setType = 2;
			}

			/// Core test set
			if (!speaker.compare("FELC0")			/// New England, female
					|| !speaker.compare("MDAB0")	/// male
					|| !speaker.compare("MWBT0")	/// male
					|| !speaker.compare("MTAS1")	/// Northern, male
					|| !speaker.compare("MWEW0")	/// male
					|| !speaker.compare("FPAS0")	/// female
					|| !speaker.compare("MJMP0")	/// North Midland, male
					|| !speaker.compare("MLNT0")	/// male
					|| !speaker.compare("FPKT0")	/// female
					|| !speaker.compare("MLLL0")	/// South Midland, male
					|| !speaker.compare("MLNT0")	/// male
					|| !speaker.compare("FPKT0")	/// female
					|| !speaker.compare("MBPM0")	/// Southern, male
					|| !speaker.compare("MKLT0")	/// male
					|| !speaker.compare("FNLP0")	/// female
					|| !speaker.compare("MCMJ0")	/// New York City, male
					|| !speaker.compare("MJDH0")	/// male
					|| !speaker.compare("FMGD0")	/// female
					|| !speaker.compare("MGRT0")	/// Western, male
					|| !speaker.compare("MNJM0")	/// male
					|| !speaker.compare("FDHC0")	/// female
					|| !speaker.compare("MJLN0")	/// Army Brat, male
					|| !speaker.compare("MPAM0")	/// male
					|| !speaker.compare("FMLD0")	/// female
					)
			{
				//cout << "Speaker: " << speaker << endl;
				coreTestCount++;
				setType = 3;
			}

			if (setType == 0)
			{
				trainCount++;
				setType = 1;
			}

			///declare and load mfcc matrix
			vector<vector<double>> mMfccCoeffs;
			render::get_mfcc_from_file(mMfccCoeffs, mfccFilename);

			///get corresponding textGrid item
			Textgrid tg(textGridFilename.c_str());
			item_c tgItem = tg.get_item(1);

			vector<double> inputVals, targetVals;

			int i = 0;
			int frame = 0;
			double frameEnd = tgItem.interval[0].xmax;
			bool done = false;

			do {
				inputVals.clear();
				inputVals.insert(inputVals.end(), mMfccCoeffs[i].begin(), mMfccCoeffs[i].end());
				//helper::print_vector("in:", inputVals);

				helper::get_textGrid_targetVals_vc(tgItem, frame, targetVals);
				//helper::print_vector("out:", targetVals);

				string outputFilename;

				if (setType == 1)
					outputFilename = "./data/set/training.set";
				else if (setType == 2)
					outputFilename = "./data/set/devTest.set";
				else if (setType == 3)
					outputFilename = "./data/set/coreTest.set";

				///	construct ofstream object and initialze filename
				ofstream outputFile(outputFilename, std::ios_base::app | std::ios_base::out);

				outputFile << "in:";
				for (int n = 0; n < inputVals.size(); n++)
				{
					outputFile << " " << inputVals.at(n);
				}
				
				outputFile << endl << "out:";
				for (int n = 0; n < targetVals.size(); n++)
				{
					outputFile << " " << targetVals.at(n);
				}
				outputFile << endl;

				outputFile.close();

				if (i == mMfccCoeffs.size() - 1)
					done = true;

				helper::get_textGrid_frame(tgItem, i, frame, frameEnd, mMfccCoeffs.size());

				i++;
			} while (!done);
		}

		j++;
		if(j >= mfccFileList.size())
			finished = true;
	}
	
	cout << "SA: " << countSA << endl
		<< "SX: " << countSX << endl
		<< "SI: " << countSI << endl
		<< "Total: " << countSA + countSX + countSI << endl;
	cout << "==============================================" << endl
		<< "Core Test Set: " << coreTestCount << endl
		<< "Dev Test Set: " << devTestCount << endl
		<< "Train Set: " << trainCount << endl;

	return 0;
}
