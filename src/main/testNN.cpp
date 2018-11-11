#include <string.h>
#include <vector>

#include "helper.h"
#include "render.h"
#include "blstm.h"
#include "dataset.h"

using namespace std;

int main(int argc, char* argv[])
{
	//string devTestFilename = "./data/set/coreTest.set";
	string devTestFilename = "./data/set/devTest.set";
	string nnFilename = "./data/nn/data.bin";

	Blstm nn(nnFilename);

	int T = nn.get_T();
	vector<unsigned> topology = nn.get_topo();
	vector<vector<double>> X;
	vector<vector<double>> Y;
	vector<vector<double>> AP;
	
	vector<vector<double>> accIter;
	vector<vector<double>> fScoreIter;

	int iterations = 1000;
	int steps = 1;
	
	DataSet devTest(devTestFilename);
	devTest.init_set(T, topology, X, Y);
	
	vector<vector<double>> confMat(Y.at(0).size(), vector<double>(Y.at(0).size(), 0.0));

	Statistics stats(Y.at(0).size());
	
	for (int iter = 0; iter < iterations; iter++)
	{
		nn.feed_forward(X);
		nn.feed_backward(X);
		nn.calculate_predictions();

		vector<vector<double>> P = nn.get_predictions();
		stats.process(Y, P);
		stats.concat_AP();
		vector<double> acc = stats.get_acc();
		vector<double> fScore = stats.get_fScore();
		accIter.push_back(acc);
		fScoreIter.push_back(fScore);
		/*std::cout << std::fixed;
		std::cout << std::setprecision(5);
		cout << "Iter: (" << iter << "|" << iterations << ")" << "\tfScore: " << fScore.at(0)
			<< "\t" << fScore.at(1) << "\t" << fScore.at(2) << "\t||\tacc: " << acc.at(0)
			<< "\t" << acc.at(1) << "\t" << acc.at(2) << endl;*/
		//stats.print_all();

		devTest.shift_set(steps, X, Y);
	}

	AP = stats.get_AP();
	render::matrix_to_file(AP, "test.AP");

	//double classErrorAvg = accumulate( classErrorIter.begin(), classErrorIter.end(), 0.0) / classErrorIter.size();
	//classErrorEpoch.push_back(classErrorAvg);
	//classErrorIter.clear();
	vector<double> fScoreAvg(fScoreIter.at(0).size(), 0.0);
	vector<double> accAvg(accIter.at(0).size(), 0.0);
	for (int c = 0; c < fScoreIter.at(0).size(); c++)
	{
		for (int t = 0; t < fScoreIter.size(); t++)
		{
			fScoreAvg.at(c) += fScoreIter.at(t).at(c);
			accAvg.at(c) += accIter.at(t).at(c);
		}
		fScoreAvg.at(c) = fScoreAvg.at(c) / fScoreIter.size();
		accAvg.at(c) = accAvg.at(c) / accIter.size();
	}

	fScoreIter.clear();
	accIter.clear();
	
	//render::vector_to_file(classErrorEpoch, "test.classError");
	//render::vector_to_file(fScoreEpoch, "test.fScore");

	//cout << "Avg. Class Error: " << classErrorAvg << endl;
	//cout << "Avg. F Score: " << fScoreAvg << endl;
	helper::print_vector("fScore: ", fScoreAvg);
	helper::print_vector("acc: ", accAvg);


	return 0;
}
