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
	
	_labelType = type;

	if ( !type.compare("vc") )
	{
		_labels.at(0) = "sil";
		_labels.at(1) = "c";
		_labels.at(2) = "v";
	} else if ( !type.compare("phn") )
	{
		_labels.at(0) = "iy";
		_labels.at(1) = "ih";
		_labels.at(2) = "eh";
		_labels.at(3) = "ae";
		_labels.at(4) = "ah";
		_labels.at(5) = "uw";
		_labels.at(6) = "uh";
		_labels.at(7) = "aa";
		_labels.at(8) = "ey";
		_labels.at(9) = "ay";
		_labels.at(10) = "oy";
		_labels.at(11) = "aw";
		_labels.at(12) = "ow";
		_labels.at(13) = "er";
		_labels.at(14) = "l";
		_labels.at(15) = "r";
		_labels.at(16) = "w";
		_labels.at(17) = "y";
		_labels.at(18) = "m";
		_labels.at(19) = "n";
		_labels.at(20) = "ng";
		_labels.at(21) = "dx";
		_labels.at(22) = "jh";
		_labels.at(23) = "ch";
		_labels.at(24) = "z";
		_labels.at(25) = "s";
		_labels.at(26) = "sh";
		_labels.at(27) = "hh";
		_labels.at(28) = "v";
		_labels.at(29) = "f";
		_labels.at(30) = "dh";
		_labels.at(31) = "th";
		_labels.at(32) = "b";
		_labels.at(33) = "p";
		_labels.at(34) = "d";
		_labels.at(35) = "t";
		_labels.at(36) = "g";
		_labels.at(37) = "k";
		_labels.at(38) = "cl";
	} else if ( !type.compare("art") )
	{
		_labels.at(0) = "vow";
		_labels.at(1) = "nas";
		_labels.at(2) = "sfr";
		_labels.at(3) = "wfr";
		_labels.at(4) = "stops";
		_labels.at(5) = "cl";
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

string Statistics::get_string_representation_vc(vector<double> binIn)
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

string Statistics::get_string_representation_phn(vector<double> binIn)
{
	string strLabel;

	vector<double> phn_iy = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ih = {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_eh = {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ae = {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ah = {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_uw = {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_uh = {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_aa = {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ey = {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ay = {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_oy = {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_aw = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ow = {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_er = {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_l =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_r =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_w =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_y =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_m =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_n =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ng = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_dx = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_jh = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_ch = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_z =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_s =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_sh = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_hh = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_v =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0};
	vector<double> phn_f =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0};
	vector<double> phn_dh = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
	vector<double> phn_th = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0};
	vector<double> phn_b =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0};
	vector<double> phn_p =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0};
	vector<double> phn_d =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0};
	vector<double> phn_t =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0};
	vector<double> phn_g =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0};
	vector<double> phn_k =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0};
	vector<double> phn_cl = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};

	if ( std::equal(binIn.begin(), binIn.end(), phn_iy.begin()) )
		strLabel = "iy";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ih.begin()) )
		strLabel = "ih";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_eh.begin()) )
		strLabel = "eh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ae.begin()) )
		strLabel = "ae";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ah.begin()) )
		strLabel = "ah";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_uw.begin()) )
		strLabel = "uw";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_uh.begin()) )
		strLabel = "uh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_aa.begin()) )
		strLabel = "aa";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ey.begin()) )
		strLabel = "ey";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ay.begin()) )
		strLabel = "ay";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_oy.begin()) )
		strLabel = "oy";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_aw.begin()) )
		strLabel = "aw";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ow.begin()) )
		strLabel = "ow";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_er.begin()) )
		strLabel = "er";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_l.begin()) )
		strLabel = "l";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_r.begin()) )
		strLabel = "r";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_w.begin()) )
		strLabel = "w";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_y.begin()) )
		strLabel = "y";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_m.begin()) )
		strLabel = "m";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_n.begin()) )
		strLabel = "n";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ng.begin()) )
		strLabel = "ng";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_dx.begin()) )
		strLabel = "dx";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_jh.begin()) )
		strLabel = "jh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_ch.begin()) )
		strLabel = "ch";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_z.begin()) )
		strLabel = "z";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_s.begin()) )
		strLabel = "s";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_sh.begin()) )
		strLabel = "sh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_hh.begin()) )
		strLabel = "hh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_v.begin()) )
		strLabel = "v";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_f.begin()) )
		strLabel = "f";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_dh.begin()) )
		strLabel = "dh";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_th.begin()) )
		strLabel = "th";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_b.begin()) )
		strLabel = "b";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_p.begin()) )
		strLabel = "p";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_d.begin()) )
		strLabel = "d";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_t.begin()) )
		strLabel = "t";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_g.begin()) )
		strLabel = "g";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_k.begin()) )
		strLabel = "k";
	else if ( std::equal(binIn.begin(), binIn.end(), phn_cl.begin()) )
		strLabel = "cl";

	return strLabel;
}

string Statistics::get_string_representation_art(vector<double> binIn)
{
	string strLabel;

	vector<double> art_vow =	{1,0,0,0,0,0};
	vector<double> art_nas =	{0,1,0,0,0,0};
	vector<double> art_sfr =	{0,0,1,0,0,0};
	vector<double> art_wfr =	{0,0,0,1,0,0};
	vector<double> art_stops =	{0,0,0,0,1,0};
	vector<double> art_cl =		{0,0,0,0,0,1};

	if ( std::equal(binIn.begin(), binIn.end(), art_vow.begin()) )
		strLabel = "vow";
	else if ( std::equal(binIn.begin(), binIn.end(), art_nas.begin()) )
		strLabel = "nas";
	else if ( std::equal(binIn.begin(), binIn.end(), art_sfr.begin()) )
		strLabel = "sfr";
	else if ( std::equal(binIn.begin(), binIn.end(), art_wfr.begin()) )
		strLabel = "wfr";
	else if ( std::equal(binIn.begin(), binIn.end(), art_stops.begin()) )
		strLabel = "stops";
	else if ( std::equal(binIn.begin(), binIn.end(), art_cl.begin()) )
		strLabel = "cl";

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

	if ( !_labelType.compare("vc") )
	{
		_AString = get_string_representation_vc(_A);
		_PString = get_string_representation_vc(_P);
	} else if ( !_labelType.compare("phn") )
	{
		_AString = get_string_representation_phn(_A);
		_PString = get_string_representation_phn(_P);
	} else if ( !_labelType.compare("art") )
	{
		_AString = get_string_representation_art(_A);
		_PString = get_string_representation_art(_P);
	}

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
	//helper::print_matrix("ConMat: ", _confusionMatrix);
	
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
