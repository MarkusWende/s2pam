#include <experimental/filesystem>
#include <chrono>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <iomanip>

#include "textgrid.h"
#include "helper.h"
#include "render.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using hires_clock = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;
using namespace std::this_thread;								// sleep_for, sleep_until
using namespace std::chrono_literals;							// ns, us, ms, s, h, etc.
namespace fs = std::experimental::filesystem;

int process(string mfccFilename, string textGridFilename)
{
	cout << mfccFilename << endl << textGridFilename << endl;
	
	///	declare mfcc matrix
	vector<vector<float>> mMfccCoeffs;

	///	initialze mfcc file
	ifstream mfccFile;
	mfccFile.exceptions ( ifstream::badbit );
	mfccFile.open(mfccFilename);
		
	///	throw error if file doesnt exit
	if(mfccFile.fail())
	{
		E_ERROR("file '" << mfccFilename << "' doesnt exist.");
		exit(1);
	}
	///	safe input line from file as a string and initialize line counter
	string line;
	int counter = 0;

	///	loop over the lines in a file
	while (getline(mfccFile, line))
	{
		///	split string around spaces
		istringstream ss(line);

		///	initialize new matrix row
		mMfccCoeffs.push_back(vector<float> (0,0));

		///	traverse through all words
		float val;
		while (ss >> val)
		{
			///	read value
			/// add element to the end of the current matrix row
			mMfccCoeffs[counter].push_back(val);
		}
		
		counter++;
	}

	///	close file
	mfccFile.close();

	vector<vector<float>> mMfccCoeffsNorm(mMfccCoeffs.size(), vector<float> (mMfccCoeffs[0].size(), 0));
	helper::matrix_to_normalized_matrix(mMfccCoeffs, mMfccCoeffsNorm);
	
	helper::print_matrix(mMfccCoeffs);
	unsigned int imageHeight;
	unsigned int imageWidth;
	vector<vector<float>> mEnl(mMfccCoeffs.size(),vector<float>(1000,0));
	vector<float> v;
	helper::matrix_enlarge(mMfccCoeffs, mEnl);
	cout << mEnl.size() << "\t" << mEnl[0].size() << endl;
	helper::matrix_to_normalized_vector(mEnl, imageHeight, imageWidth, v);
	render::vector_to_PNG("blub", "_mfcc", imageHeight, imageWidth, v);

	Textgrid tg(textGridFilename.c_str());
	item_c tgItem = tg.get_item(1);
	//cout << "Name: " << tgItem.name << " || Size: " << tgItem.size << endl;
	//cout << "Xmin: " << tgItem.xmin << " || Xmax: " << tgItem.xmax << endl;

	for (int i = 0; i < tgItem.size; i++)
	{
		//cout << "Text: " << tgItem.interval[i].text << "\t\tXmin: " << tgItem.interval[i].xmin 
		//	<< "\tXmax: " << tgItem.interval[i].xmax << endl;
	}
	
	return 0;
}

int main(int argc, char* argv[])
{
	auto t1 = hires_clock::now();
	
	// check if audio files for training do exist
	if (!fs::is_directory("./data/FEATURES") || !fs::exists("./data/FEATURES"))
	{
		E_ERROR("\tNo MFCC data folder found.\n" <<
			"\t\t1. run s2pam_featureExtraction\n\t\t2. run s2pam_trainNN");
		exit(1);
	}

	// set the logging level
	if (argc > 1)
	{
		string argVerbose = "-v";
		if(argVerbose.compare(argv[0]) == 0)
			infoLevelActive = true;
		else
			infoLevelActive = false;
	}
	else
		infoLevelActive = false;

	//tg.print_textgrid_struct();

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

	essentia::init();
	string mfccFilename;
	string textGridFilename;
	bool finished = false;
	int i = 0;
	
	while(!finished)
	{	
		mfccFilename = mfccFileList[i];
		textGridFilename = textGridFileList[i];
		
		process(mfccFilename, textGridFilename);

		i++;
		if(i >= mfccFileList.size())
			finished = true;

		//sleep_for(500ms);	
	
		//return 0;
	}

	essentia::shutdown();
	std::cout << "Elapsed: " << duration_ms(hires_clock::now() - t1).count() << " ms\n";
	
	return 0;
}
