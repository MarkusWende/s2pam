/**
 * @file	statistic.cpp
 *
 * @brief	Collection of functions for statistic analysing
 *
 *			This namespace contains functions for matrix manipulation
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "statistic.h"

using namespace std;
using namespace helper;

Statistics::Statistics(int numClass)
{
	_confusionMatrix.clear();
	_confusionMatrix.resize(numClass, vector<double>(numClass, 0.0));

	_fScore.clear();
	_fScore.resize(numClass, 0.0);
	
	_precision.clear();
	_precision.resize(numClass, 0.0);

	_recall.clear();
	_recall.resize(numClass, 0.0);

	_acc.clear();
	_acc.resize(numClass, 0.0);
	
	_classError.clear();
	_classError.resize(numClass, 0.0);
	
	_TP.clear();
	_TP.resize(numClass, 0.0);
	
	_TN.clear();
	_TN.resize(numClass, 0.0);
	
	_FP.clear();
	_FP.resize(numClass, 0.0);
	
	_FN.clear();
	_FN.resize(numClass, 0.0);
	
	_total.clear();
	_total.resize(numClass, 0.0);

	_fScoreSum = 0.0;
	_accSum = 0.0;
}

void Statistics::true_positive()
{
	for (int c = 0; c < _TP.size(); c++)
	{
		double TP = 0.0;

		for (int i = 0; i < _A.size(); i++)
		{
			if ( (_A.at(i).at(c) == 1.0) && (_P.at(i).at(c) == 1.0) )
				TP += 1.0;
		}

		_TP.at(c) = TP;
	}
}

void Statistics::true_negative()
{
	for (int c = 0; c < _TN.size(); c++)
	{
		double TN = 0.0;

		for (int i = 0; i < _A.size(); i++)
		{
			if ( (_A.at(i).at(c) == 0.0) && (_P.at(i).at(c) == 0.0) )
				TN += 1.0;
		}

		_TN.at(c) = TN;
	}
}

void Statistics::false_positive()
{
	for (int c = 0; c < _FP.size(); c++)
	{
		double FP = 0.0;

		for (int i = 0; i < _A.size(); i++)
		{
			if ( (_A.at(i).at(c) == 0.0) && (_P.at(i).at(c) == 1.0) )
				FP += 1.0;
		}

		_FP.at(c) = FP;
	}
}

void Statistics::false_negative()
{
	for (int c = 0; c < _FN.size(); c++)
	{
		double FN = 0.0;

		for (int i = 0; i < _A.size(); i++)
		{
			if ( (_A.at(i).at(c) == 1.0) && (_P.at(i).at(c) == 0.0) )
				FN += 1.0;
		}

		_FN.at(c) = FN;
	}
}

void Statistics::precision()
{
	for (int c = 0; c < _precision.size(); c++)
	{
		double precision = 0.0;

		if (_TP.at(c) == 0)
			precision = 0.0;
		else
			precision = _TP.at(c) / ( _TP.at(c) + _FP.at(c) );

		_precision.at(c) = precision;
	}
}

void Statistics::recall()
{
	for (int c = 0; c < _recall.size(); c++)
	{
		double recall = 0.0;

		if (_TP.at(c) == 0)
			recall = 0.0;
		else
			recall = _TP.at(c) / ( _TP.at(c) + _FN.at(c) );

		_recall.at(c) = recall;
	}
}

void Statistics::accuracy()
{
	double accSum = 0.0;
	double totalSum = 0.0;

	for (int c = 0; c < _acc.size(); c++)
	{
		double acc = 0.0;
		double hit = 0.0;

		_total.at(c) = (_TP.at(c) + _TN.at(c) + _FP.at(c) + _FN.at(c) );
		hit = ( _TP.at(c) + _TN.at(c) );
		accSum += hit;
		totalSum += _total.at(c);

		acc = hit / _total.at(c);
		_acc.at(c) = acc * 100;
	}

	_accSum = accSum / totalSum * 100;
}

void Statistics::fScore()
{
	double fScoreSum = 0.0;
	double precisionSum = 0.0;
	double recallSum = 0.0;
	
	for (int c = 0; c < _fScore.size(); c++)
	{
		double fScore = 0.0;
		
		if ( (_precision.at(c) != 0) && (_recall.at(c) != 0) )
			fScore = 2 * _precision.at(c) * _recall.at(c) / (_precision.at(c) + _recall.at(c));
		
		precisionSum += _precision.at(c);
		recallSum += _recall.at(c);

		_fScore.at(c) = fScore;
	}
	
	if ( (precisionSum != 0) && (recallSum != 0) )
		fScoreSum = 2 * precisionSum * recallSum / (precisionSum + recallSum);
	
	_fScoreSum = fScoreSum;
}

void Statistics::fScore_sum()
{

	for (int c = 0; c < _fScore.size(); c++)
	{
	}

}

void Statistics::confusion_matrix()
{
	for (int i = 0; i < _A.at(0).size(); i++)
	{
		for (int j = 0; j < _A.at(0).size(); j++)
		{
			double counter = 0.0;
			for (int t = 0; t < _A.size(); t++)
			{
				if ( (_A.at(t).at(i) == 1.0) && (_P.at(t).at(j) == 1.0) )
				{
					counter = counter + 1.0;
				}
			}
			_confusionMatrix.at(i).at(j) = counter;
		}
	}
}

void Statistics::process(vector<vector<double>> A, vector<vector<double>> P)
{
	_A = A;

	for (int t = 0; t < P.size(); t++)
	{
		_P.push_back( get_oneHot(P.at(t)) );
	}

	confusion_matrix();

	true_positive();
	true_negative();
	false_positive();
	false_negative();

	//_total += accumulate( _TP.begin(), _TP.end(), 0.0);

	precision();
	recall();
	fScore();
	accuracy();
}

void Statistics::print_all()
{
	helper::print_matrix("ConMat: ", _confusionMatrix);
	
	for (int c = 0; c < _TP.size(); c++)
	{
		cout << "-------------------------------------" << endl;
		cout << "Class: " << c << endl;
		cout << "TP: " << _TP.at(c) << "\tTN: " << _TN.at(c) << "\tFP: " << _FP.at(c) << "\tFN: " << _FN.at(c) << endl;
		cout << "fScore: " << _fScore.at(c) << "\tAcc: " << _acc.at(c) << "\tPrecision: " << _precision.at(c) << endl << endl;
	}

	cout << "====================================" << endl;
	cout << "FScore Sum: " << _fScoreSum << "\tAcc Sum: " << _accSum << endl;
}
