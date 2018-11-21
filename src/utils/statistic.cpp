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

Statistics::Statistics(int numClass, string type, double p)
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

	_accSum = 0.0;
	_population = p;

	_labels.clear();
	_labels.resize(numClass, "null");

	if ( !type.compare("vc") )
	{
		_labels.at(0) = "sil";
		_labels.at(1) = "c";
		_labels.at(2) = "v";
	}
}

void Statistics::true_positive()
{
	for (int c = 0; c < _TP.size(); c++)
	{
		if ( (_A.at(c) == 1.0) && (_P.at(c) == 1.0) )
			_TP.at(c) += 1.0;
	}
}

void Statistics::true_negative()
{
	for (int c = 0; c < _TN.size(); c++)
	{
		if ( (_A.at(c) == 0.0) && (_P.at(c) == 0.0) )
			_TN.at(c) += 1.0;
	}
}

void Statistics::false_positive()
{
	for (int c = 0; c < _FP.size(); c++)
	{
		if ( (_A.at(c) == 0.0) && (_P.at(c) == 1.0) )
			_FP.at(c) += 1.0;
	}
}

void Statistics::false_negative()
{
	for (int c = 0; c < _FN.size(); c++)
	{
		if ( (_A.at(c) == 1.0) && (_P.at(c) == 0.0) )
			_FN.at(c) += 1.0;
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
		
		//cout << "TP: " << _TP.at(c) << "\tTN: " << _TN.at(c) << "\tFP: " << _FP.at(c) << "\tFN: " << _FN.at(c) << endl;

		_total.at(c) = (_TP.at(c) + _TN.at(c) + _FP.at(c) + _FN.at(c) );
		hit = ( _TP.at(c) + _TN.at(c) );
		accSum += hit;
		totalSum += _total.at(c);

		acc = hit / _total.at(c);
		_acc.at(c) = acc * 100;
		_accSum += _TP.at(c);
	}
	_accSum /= _population;
}

void Statistics::fScore()
{
	double precisionSum = 0.0;
	double recallSum = 0.0;
	
	for (int c = 0; c < _fScore.size(); c++)
	{
		double fScore = 0.0;
		
		if ( (_precision.at(c) != 0) && (_recall.at(c) != 0) )
			fScore = 2 * _precision.at(c) * _recall.at(c) / (_precision.at(c) + _recall.at(c));

		_fScore.at(c) = fScore;
	}
}

void Statistics::confusion_matrix()
{
	for (int i = 0; i < _A.size(); i++)
	{
		for (int j = 0; j < _A.size(); j++)
		{
			if ( (_A.at(i) == 1.0) && (_P.at(j) == 1.0) )
			{
				_confusionMatrix.at(i).at(j) += 1;
			}
		}
	}
}

string Statistics::get_string_representation(vector<double> binIn)
{
	string strLabel;

	vector<double> vc_sil = {1,0,0};
	vector<double> vc_c = {0,1,0};
	vector<double> vc_v = {0,0,1};

	if ( std::equal(binIn.begin(), binIn.end(), vc_sil.begin()) )
		strLabel = "sil";
	else if ( std::equal(binIn.begin(), binIn.end(), vc_c.begin()) )
		strLabel = "c";
	else if ( std::equal(binIn.begin(), binIn.end(), vc_v.begin()) )
		strLabel = "v";

	return strLabel;
}

void Statistics::concat_AP_binary()
{
	vector<double> ap = _A;
	ap.insert( ap.end(), _P.begin(), _P.end() );
	_AP.push_back( ap );
}

void Statistics::concat_AP()
{
	vector<string> ap;
	ap.push_back( _AString );
	ap.push_back( _PString );
	_APString.push_back( ap );
}

void Statistics::process(vector<double> A, vector<double> P)
{
	_A = A;
	_P = get_oneHot( P );
	_AString = get_string_representation(_A);
	_PString = get_string_representation(_P);

	concat_AP_binary();
	concat_AP();

	//helper::print_2matrices_column("A and P: ", _A, _P);

	confusion_matrix();

	true_positive();
	true_negative();
	false_positive();
	false_negative();

	//_total += accumulate( _TP.begin(), _TP.end(), 0.0);

}

void Statistics::print_all()
{
	helper::print_matrix("ConMat: ", _confusionMatrix);
	
	precision();
	recall();
	fScore();
	accuracy();
	
	for (int c = 0; c < _TP.size(); c++)
	{
		cout << "-------------------------------------" << endl;
		cout << "Class No: " << c << "\tLabel: " << _labels.at(c) << endl;
		cout << "TP: " << _TP.at(c) << "\tTN: " << _TN.at(c) << "\tFP: " << _FP.at(c) << "\tFN: " << _FN.at(c) << endl;
		cout << "fScore: " << _fScore.at(c) << "\tAcc: " << _acc.at(c) << "\tPrecision: " << _precision.at(c) << endl << endl;
	}

	cout << "====================================" << endl;
	cout << "Acc Sum: " << _accSum << endl;
}
