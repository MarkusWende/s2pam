#include <experimental/filesystem>
#include <chrono>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <iomanip>
#include <iterator>
#include <numeric>							/// std::accumulate

#include "textgrid.h"
#include "helper.h"
#include "render.h"
#include "blstm.h"
#include "dataset.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using hires_clock = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;
using namespace std::this_thread;								// sleep_for, sleep_until
using namespace std::chrono_literals;							// ns, us, ms, s, h, etc.
namespace fs = std::experimental::filesystem;

void process()
{
	//TrainingData trainData("data/AND/testBIG.txt");
	//TrainingData trainData("data/test/test01.txt");
	//TrainingData trainData("data/AND/T241L20000.txt");
	//TrainingData trainData("data/AND/vctest.txt");
	string trainFilename = "./data/set/training.set";
	string devTestFilename = "./data/set/devTest.set";
	//TrainingData trainData("data/NAND/T241L20000.txt");
	// e.g., { 3, 2, 1 }
	vector<unsigned> topology = {39, 60, 3};
	int T = 200;
	int maxEpoch = 50;
	int steps = 180;
	double learningRate = 0.0001;

	vector<vector<double>> X;
	vector<vector<double>> Y;

	Blstm nn(topology, T, learningRate);
	nn.random_weights();
	
	double results;
	int trainingSize = 0;
	int testSize = 0;
	int epoch = 0;

	vector<double> errorEpoch;
	vector<double> errorIter;

	bool done = false;	
	
	DataSet train(trainFilename);
	train.init_set(T, topology, X, Y);
	//int iterations = train.size() - T;
	int iterations = 1000;

	//cout << "Size: " << iterations << endl;
	//return;

	//helper::print_2matrices_column("X and Y", X, Y);

	//return;

	do {
		epoch++;

		for (int iter = 0; iter < iterations; iter++)
		{
			nn.feed_forward(X);
			nn.feed_backward(X);
			nn.calculate_predictions();

			nn.bptt(X,Y);
			nn.fptt(X,Y);

			double L = nn.calculate_loss(Y);
			errorIter.push_back(L);
			cout << "Epoch: (" << epoch << "|" << maxEpoch << ")\tIter: (" << iter
				<< "|" << iterations << ")\tLoss: " << L << endl;

			//helper::print_matrix("Y", Y);
			train.shift_set(steps, X, Y);
			//helper::print_matrix("Y", Y);
			//exit(0);
		}

		double errorAvg = accumulate( errorIter.begin(), errorIter.end(), 0.0) / errorIter.size();
		errorEpoch.push_back(errorAvg);
		errorIter.clear();

		train.return_to_begin_of_file();

		//if ( !nn.check_weight_sum() )
		//	done = true;
		
		nn.render_weights(epoch);

		if (epoch == maxEpoch)
		{
			//nn.print_result(Y);
			done = true;
		}

	} while (!done);

	render::vector_to_file(errorEpoch, "epoch.error");
	
	/// Testing
	DataSet devTest(devTestFilename);
	devTest.init_set(T, topology, X, Y);

	nn.feed_forward(X);
	nn.feed_backward(X);
	nn.calculate_predictions();
	nn.print_result(Y);

	double L = nn.calculate_loss(Y);
	cout << "Loss: " << L << endl;;
	nn.save();
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
