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
	string nnFilename = "./data/nn/data_vc4.bin";

	Blstm nn(nnFilename);

	int T = nn.get_T();
	vector<unsigned> topology = nn.get_topo();
	vector<vector<double>> X;
	vector<vector<double>> Y;
	
	vector<double> classErrorEpoch;
	vector<double> classErrorIter;
	vector<double> fScoreEpoch;
	vector<double> fScoreIter;

	int iterations = 10;
	int steps = T;
	
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
		//double classError = statistic::get_avg_classification_error(Y, P);
		//double f_score = statistic::get_avg_fScore(Y, P);
		//if (!isnan(classError))
		//	classErrorIter.push_back(classError);
		//if (!isnan(f_score))
		//	fScoreIter.push_back(f_score);
		//cout << "Iter: (" << iter << "|" << iterations << ")\tAvgClassError: " << classError
		//	<< "\tAvgFScore: " << f_score << endl;
		//confMat = helper::matrix_add(confMat, statistic::get_confusion_matrix(Y, P));
		stats.print_all();

		devTest.shift_set(steps, X, Y);
	}
/*
	double classErrorAvg = accumulate( classErrorIter.begin(), classErrorIter.end(), 0.0) / classErrorIter.size();
	classErrorEpoch.push_back(classErrorAvg);
	classErrorIter.clear();
	double fScoreAvg = accumulate( fScoreIter.begin(), fScoreIter.end(), 0.0) / fScoreIter.size();
	fScoreEpoch.push_back(fScoreAvg);
	fScoreIter.clear();
	
	render::vector_to_file(classErrorEpoch, "test.classError");
	render::vector_to_file(fScoreEpoch, "test.fScore");

	cout << "Avg. Class Error: " << classErrorAvg << endl;
	cout << "Avg. F Score: " << fScoreAvg << endl;
	helper::print_matrix("ConfMatrix: ", confMat);
*/	

	return 0;
}
