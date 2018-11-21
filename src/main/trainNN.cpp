#include <experimental/filesystem>
#include <chrono>
//#include <essentia/algorithmfactory.h>
//#include <essentia/essentiamath.h>
//#include <essentia/pool.h>
#include <iomanip>
#include <iterator>
#include <numeric>							/// std::accumulate

#include "textgrid.h"
#include "helper.h"
#include "render.h"
#include "blstm.h"
#include "dataset.h"
#include "statistic.h"

using namespace std;
//using namespace essentia;
//using namespace essentia::standard;
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
	//TrainingData trainData("data/NAND/T241L20000.txt");
	// e.g., { 3, 2, 1 }
	vector<unsigned> topology = {39, 60, 3};
	int T = 100;
	int maxEpoch = 20;
	int steps = 1;
	double learningRate = 0.001;

	vector<vector<double>> X;
	vector<vector<double>> Y;
	vector<vector<double>> bX;
	vector<vector<double>> bY;

	Blstm nn(topology, T, learningRate);
	nn.random_weights();
	
	double results;
	int trainingSize = 0;
	int testSize = 0;
	int epoch = 0;

	vector<double> lossEpoch;
	vector<double> lossIter;
	vector<double> classErrorEpoch;
	vector<double> classErrorIter;
	vector<double> fScoreEpoch;
	vector<double> fScoreIter;

	bool done = false;	
	
	DataSet train(trainFilename);
	train.init_set(T, topology, X, Y, bX, bY);
	//int iterations = train.size() - T;
	int iterations = 10000;

	//cout << "Size: " << iterations << endl;
	//return;

	//helper::print_2matrices_column("X and Y", X, Y);

	//return;

	do {
		epoch++;

		for (int iter = 0; iter < iterations; iter++)
		{
			nn.feed_forward(X);
			nn.feed_backward(bX);
			nn.calculate_single_predictions();

			vector<double> target(Y.at(0).size(), 0.0);
			for (int c = 0; c < target.size(); c++)
			{
				target.at(c) = Y.at(T-1).at(c);
			}

			nn.bptt(X,Y,target);
			nn.fptt(bX,bY,target);

			double L = nn.calculate_single_loss(target);
			if (!isnan(L))
				lossIter.push_back(L);
			cout << "Epoch: (" << epoch << "|" << maxEpoch << ")\tIter: (" << iter
				<< "|" << iterations << ")\tLoss: " << L << endl;

			train.shift_set(steps, X, Y, bX, bY);
		}

		double lossAvg = accumulate( lossIter.begin(), lossIter.end(), 0.0) / lossIter.size();
		lossEpoch.push_back(lossAvg);
		lossIter.clear();

		train.return_to_begin_of_file();

		nn.render_weights(epoch);

		if (epoch == maxEpoch)
		{
			//nn.print_result(Y);
			done = true;
		}

	} while (!done);

	render::vector_to_file(lossEpoch, "train.loss");
	
	nn.save();
}

int main(int argc, char* argv[])
{
	auto t1 = hires_clock::now();

	// set the logging level
	if (argc > 1)
	{
		string argVerbose = "-v";
	}

	process();
	
	std::cout << "Elapsed: " << duration_ms(hires_clock::now() - t1).count() << " ms\n";
	
	return 0;
}
