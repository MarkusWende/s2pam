#include <experimental/filesystem>
#include <chrono>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <iomanip>

#include <iterator>

#include "textgrid.h"
#include "helper.h"
#include "render.h"
#include "blstm.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using hires_clock = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;
using namespace std::this_thread;								// sleep_for, sleep_until
using namespace std::chrono_literals;							// ns, us, ms, s, h, etc.
namespace fs = std::experimental::filesystem;


class TrainingData
{
	public:
		TrainingData(const string filename);
		bool isEof() { return m_trainingDataFile.eof(); };
		void getTopology(vector<unsigned> &topology);
		unsigned getFileLength(void);

		// Returns the number of input values read from the file:
		unsigned getNextInputs(vector<double> &inputVals);
		unsigned getTargetOutputs(vector<double> &targetOutputVals);

	private:
		ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") == 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainingData::getFileLength(void)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("length:") == 0) {
		abort();
	}

	unsigned n;
	while (!ss.eof()) {
		ss >> n;
	}

	return n;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

int process_test()
{
	TrainingData trainData("data/AND/test2.txt");
	//TrainingData trainData("data/AND/T241L20000.txt");
	// e.g., { 3, 2, 1 }
	vector<unsigned> topology;
	trainData.getTopology(topology);
	Blstm myNet(topology);

	unsigned fileLength = trainData.getFileLength();
	
	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	bool done = false;	

	 do {
		cout << trainingPass << ",";
		++trainingPass;

		// Get new input data feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		myNet.feed_forward(inputVals);

		// Collect the net's actual results:
		myNet.get_results(resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		assert(targetVals.size() == topology.back());

		myNet.back_prop(targetVals);
		cout << myNet.get_recent_average_error() << endl;

		// Report how well the training is working, averaged over recent 
		if (trainingPass == fileLength + 1) {
		/*	cout << endl << "Pass " << trainingPass;
			helper::print_vector("Inputs:", inputVals);
			helper::print_vector("Outputs:", resultVals);
			helper::print_vector("Targets:", targetVals);
			cout << "Net recent average error: "
				<< fixed << myNet.get_recent_average_error() << endl;

			helper::print_neural_network_graph(myNet);
*/
			done = true;
		}

	} while (!done);

	vector<double> input;
	vector<float> results;
	vector<float> targets;
	int testPass = 0;

	done = false;	

	 do {
		++testPass;

		// Get new input data feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		myNet.feed_forward(inputVals);

		// Collect the net's actual results:
		myNet.get_results(resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		assert(targetVals.size() == topology.back());

		//helper::print_vector("Result:", resultVals);
		//helper::print_vector("Target:", targetVals);
		results.push_back(resultVals.at(0));
		targets.push_back(targetVals.at(0));

		// Report how well the training is working, averaged over recent 
		if (testPass == 200) {
	/*		cout << endl << "Pass " << testPass;
			helper::print_vector("Inputs:", inputVals);
			helper::print_vector("Outputs:", resultVals);
			helper::print_vector("Targets:", targetVals);
			cout << "Net recent average error: "
				<< fixed << myNet.get_recent_average_error() << endl;

			helper::print_neural_network_graph(myNet);
*/
			done = true;
		}

	} while (!done);
	
	render::vector_to_file(results, "results");
	render::vector_to_file(targets, "targets");
}

int process()
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
	
	/// initliaze topology of the NN
	/// e.g., { 13, 24, 3 } => 13 inputs, 24 cells and 3 outputs
	vector<unsigned> topology;
	topology.push_back(13);
	topology.push_back(26);
	topology.push_back(1);

	///	create NN with topology
	Blstm nn(topology);
	
	string mfccFilename;
	string textGridFilename;
	bool finished = false;
	int j = 0;

	/// Main Loop over all mfcc and textGrid files in the corresponding folders
	while(!finished)
	{	
		mfccFilename = mfccFileList[j];
		textGridFilename = textGridFileList[j];

		///	declare and load mfcc matrix
		vector<vector<float>> mMfccCoeffs;
		render::get_mfcc_from_file(mMfccCoeffs, mfccFilename);

		///	get corresponding textGrid item
		Textgrid tg(textGridFilename.c_str());
		item_c tgItem = tg.get_item(1);

		///
		vector<double> inputVals, targetVals, resultVals;
		int i = 0;
		int frame = 0;
		float frameEnd = tgItem.interval[0].xmax;
		bool done = false;

		do {
			inputVals.clear();
			inputVals.insert(inputVals.end(), mMfccCoeffs[i].begin(), mMfccCoeffs[i].end());
			//cout << "Text: " << tgItem.interval[frame].text << endl;
			helper::print_vector("in:", inputVals);
			nn.feed_forward(inputVals);

			/// Collect the net's actual results
			nn.get_results(resultVals);

			/// get textGrid target Values of dimension 3 => e.g {1,0,0} = "sil"; {0,1,0} = "c"; 
			/// {0,0,1} ="v"
			helper::get_textGrid_targetVals_vc(tgItem, frame, targetVals);

			//cout << "Text: " << tgItem.interval[frame].text << endl;
			//helper::print_vector("Val: ", targetVals);
			assert(targetVals.size() == topology.back());

			nn.back_prop(targetVals);

			// Report how well the training is working, averaged over recent 

			if (i == mMfccCoeffs.size() - 1)
			{
				done = true;
			}

			helper::get_textGrid_frame(tgItem, i, frame, frameEnd, mMfccCoeffs.size());

			i++;
		} while (!done);
		
		j++;
		if(j >= mfccFileList.size())
			finished = true;

		//sleep_for(500ms);	
	
		//return 0;
	}


/*
	vector<double> input;
	// Text: sil
	// 0.743879 0.741406 0.747394 0.744390 0.745107 0.749080 0.741619 0.741712 0.750194 0.745640 0.742575
	// 0.746821 0.742977
	input.clear();
	input.push_back(0.743879);
	input.push_back(0.741406);
	input.push_back(0.747394);
	input.push_back(0.744390);
	input.push_back(0.745107);
	input.push_back(0.749080);
	input.push_back(0.741619);
	input.push_back(0.741712);
	input.push_back(0.750194);
	input.push_back(0.745640);
	input.push_back(0.742575);
	input.push_back(0.746821);
	input.push_back(0.742977);
*/
/*	// Text: v
	// 0.809542 0.167277 0.701892 0.718764 0.544742 0.716118 0.682328 0.716513 0.850773 0.788509 0.758259
	// 0.845399 0.828162
	input.clear();
	input.push_back(0.809542);
	input.push_back(0.167277);
	input.push_back(0.701892);
	input.push_back(0.718764);
	input.push_back(0.544742);
	input.push_back(0.716118);
	input.push_back(0.682328);
	input.push_back(0.716513);
	input.push_back(0.850773);
	input.push_back(0.788509);
	input.push_back(0.758259);
	input.push_back(0.845399);
	input.push_back(0.828162);
*/
/*	nn.feed_forward(input);

	// Collect the net's actual results:
	vector<double> resultVals;
	vector<double> targetVals;
	nn.get_results(resultVals);

	// v
	//targetVals.push_back(0);
	//targetVals.push_back(0);
	targetVals.push_back(1);

	nn.back_prop(targetVals);
*/	
//	helper::print_neural_network_graph(nn);
/*
	cout << "=======================================" << endl << endl;
	helper::print_vector("Input: ", input);
	helper::print_vector("Target: ", targetVals);
	helper::print_vector("Outputs:", resultVals);
*/
	
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


	essentia::init();



	//process();
	process_test();
	
	
	essentia::shutdown();
	//std::cout << "Elapsed: " << duration_ms(hires_clock::now() - t1).count() << " ms\n";
	
	return 0;
}
