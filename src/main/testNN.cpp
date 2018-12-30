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
	string devTestFilename = "./data/set/devTest_vc.set";
	//string devTestFilename = "./data/set/blub.set";
	//string devTestFilename = "./data/set/training.set";
	string nnFilename = "./data/nn/data_vc10.bin";

	Blstm nn(nnFilename);

	int T = nn.get_T();
	vector<unsigned> topology = nn.get_topo();
	vector<vector<double>> X;
	vector<vector<double>> Y;
	vector<vector<double>> bX;
	vector<vector<double>> bY;
	vector<vector<double>> AP;
	vector<vector<string>> APString;
	
	vector<vector<double>> accIter;
	vector<vector<double>> fScoreIter;

	//int iterations = 10000;
	int iterations = 400000;
	int steps = 1;
	string type = "vc";
	//string type = "phn";
	//string type = "art";

	DataSet devTest(devTestFilename);
	devTest.init_set(T, topology, X, Y, bX, bY);
	//devTest.init_set(10, {3, 10, 3}, X, Y, bX, bY);
	
	//helper::print_2matrices_column("Y und bY", Y, bY);
	//devTest.shift_set(steps, X, Y, bX, bY);
	//helper::print_2matrices_column("Y und bY", Y, bY);
	//exit(0);
	
	vector<vector<double>> confMat(Y.at(0).size(), vector<double>(Y.at(0).size(), 0.0));

	Statistics stats(Y.at(0).size(), type, iterations);
	
	for (int iter = 0; iter < iterations; iter++)
	{
		nn.feed_forward(X);
		nn.feed_backward(bX);
		nn.calculate_single_predictions();

		//vector<vector<double>> P = nn.get_predictions();
		vector<double> p = nn.get_single_prediction();
		//stats.process(Y, P);
		vector<double> target = Y.at(T-1);
		//helper::print_vector("Target: ", target);
		//helper::print_vector("bY.at(0): ", bY.at(0));
		stats.process(target, p);
		//vector<double> acc = stats.get_acc();
		//vector<double> fScore = stats.get_fScore();
		//accIter.push_back(acc);
		//fScoreIter.push_back(fScore);
		//std::cout << std::fixed;
		//std::cout << std::setprecision(5);
		cout << "Iter: (" << iter << "|" << iterations << ")" << endl; /*"\tfScore: " << fScore.at(0)
			<< "\t" << fScore.at(1) << "\t" << fScore.at(2) << "\t||\tacc: " << acc.at(0)
			<< "\t" << acc.at(1) << "\t" << acc.at(2) << endl;*/

/*		vector<double> pOH;
		pOH = helper::get_oneHot(p);
		cout << "Target: sil: " << target.at(0) << "\tc: " << target.at(1) << "\tv: "
			<< target.at(2) << "\t\tPrediction: sil: " << pOH.at(0) << "\tc: " << pOH.at(1)
			<< "\tv: " << pOH.at(2) << endl;
*/
		devTest.shift_set(steps, X, Y, bX, bY);
	}
	confMat = stats.get_confMat();
	AP = stats.get_AP();
	APString = stats.get_APString();
	render::matrix_to_file(confMat, "test.confMat");
	render::matrix_to_file(AP, "test.AP");
	render::matrix_to_file(APString, "test.APString");

	//double classErrorAvg = accumulate( classErrorIter.begin(), classErrorIter.end(), 0.0) / classErrorIter.size();
	//classErrorEpoch.push_back(classErrorAvg);
	//classErrorIter.clear();
	/*vector<double> fScoreAvg(fScoreIter.at(0).size(), 0.0);
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
	*/
	//render::vector_to_file(classErrorEpoch, "test.classError");
	//render::vector_to_file(fScoreEpoch, "test.fScore");

	//cout << "Avg. Class Error: " << classErrorAvg << endl;
	//cout << "Avg. F Score: " << fScoreAvg << endl;
	//helper::print_vector("fScore: ", fScoreAvg);
	//helper::print_vector("acc: ", accAvg);

	stats.print_all();

	return 0;
}
