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
		void returnToBeginOfFile();
		unsigned getFileLength(void);

		// Returns the number of input values read from the file:
		unsigned getNextInputs(vector<float> &inputVals);
		unsigned getTargetOutputs(vector<float> &targetOutputVals);

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

void TrainingData::returnToBeginOfFile()
{
	m_trainingDataFile.clear();
	m_trainingDataFile.seekg(0, std::ios::beg);
	
	string line;
	getline(m_trainingDataFile, line);
	getline(m_trainingDataFile, line);
}

unsigned TrainingData::getNextInputs(vector<float> &inputVals)
{
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		float oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<float> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		float oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

void process()
{
	//TrainingData trainData("data/AND/testBIG.txt");
	//TrainingData trainData("data/test/test01.txt");
	//TrainingData trainData("data/AND/T241L20000.txt");
	TrainingData trainData("data/AND/vctest.txt");
	//TrainingData trainData("data/NAND/T241L20000.txt");
	// e.g., { 3, 2, 1 }
	vector<unsigned> topology;
	int T = 50;
	float learningRate = 0.001;

	trainData.getTopology(topology);
	vector<vector<float>> X;
	vector<vector<float>> Y;

	Blstm nn(topology, T, learningRate);

	//nn.add_recursion();
	nn.random_weights();
	//nn.add_bias();

	//nn.print_structure();
	//return;	

	unsigned fileLength = trainData.getFileLength();
	
	vector<float> inputVals, targetVals, error;
	vector<float> resultsToFile, targetsToFile;
	float results;
	int trainingSize = 0;
	int testSize = 0;
	int epoch = 0;

	bool done = false;	
	
	for (int t = 0; t < T; t++)
	{
		// Get new input data feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			cout << "Topo[0]: " << topology[0] << "\tSize input: " << inputVals.size() << endl;
			break;
		}

		X.push_back( inputVals );

		//helper::print_matrix("X:", X);
		
		trainData.getTargetOutputs(targetVals);
		Y.push_back( targetVals );
	}

	do {
		epoch++;
		//helper::print_matrix("X", X);
		//helper::print_matrix("Y", Y);

		nn.forward_prop(X);
		float L = nn.calculate_loss(Y);

		cout << "Loss: " << L << endl;

		nn.bptt(X,Y);

		if (epoch == 600) {
			//nn.print_result(Y);
			done = true;
		}

	} while (!done);
	
	/// Testing
	X.clear();
	Y.clear();
	for (int t = 0; t < T; t++)
	{
		// Get new input data feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			cout << "Topo[0]: " << topology[0] << "\tSize input: " << inputVals.size() << endl;
			break;
		}

		X.push_back( inputVals );

		//helper::print_matrix("X:", X);
		
		trainData.getTargetOutputs(targetVals);
		Y.push_back( targetVals );
	}

	nn.forward_prop(X);
	nn.print_result(Y);

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



	process();
	
	
	essentia::shutdown();
	//std::cout << "Elapsed: " << duration_ms(hires_clock::now() - t1).count() << " ms\n";
	
	return 0;
}
