/**
 * @file	blstm.cpp
 * @class	BLSTM
 *
 * @brief	BLSTM (Bidirectional long short term memory) Neural Network
 *
 *			This class represents the blstm neural network skeletal structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "blstm.h"

using namespace std;

Blstm::Blstm(vector<unsigned> topo, int T, float lR)
{
	_T = T;
	_bpttTruncate = 4;
	_learningRate = lR;

	_iLSize = topo.at(0);
	_hLSize = topo.at(1);
	_oLSize = topo.at(2);

	_U.clear();
	_U.resize(_hLSize, vector<float> (_iLSize, 0));
	_V.clear();
	_V.resize(_oLSize, vector<float> (_hLSize, 0));
	_W.clear();
	_W.resize(_hLSize, vector<float> (_hLSize, 0));

	//helper::print_matrix("U:", _U);
	//helper::print_matrix("V:", _V);
	//helper::print_matrix("W:", _W);
}

float Blstm::tanhyp(float x)
{
	float fx = 0.0;
	fx = tanh(x);

	return fx;
}

vector<float> Blstm::softmax(vector<float> x)
{
	vector<float> softmax(x.size(), 0);
	float xExpSum = 0;
	
	for (int i = 0; i < x.size(); i++)
	{
		xExpSum += exp(x.at(i));
	}

	for (int i = 0; i < x.size(); i++)
	{
		softmax.at(i) = exp(x.at(i)) / xExpSum;
	}

	return softmax;
}

vector<vector<float>> Blstm::matrix_add(vector<vector<float>> A, vector<vector<float>> B)
{
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<float>> Blstm::matrix_add_with_const(vector<vector<float>> A, vector<vector<float>> B, float x)
{
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + x * B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<float>> Blstm::matrix_mult(vector<vector<float>> A, vector<vector<float>> B)
{
	if (A.at(0).size() != B.size())
	{
		cout << "Matrices cant be multiplied." << endl;
	}

	vector<vector<float>> out(A.size(), vector<float> (B.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < B.at(0).size(); j++)
		{
			for (int k = 0; k < B.size(); k++)
			{
				out.at(i).at(j) = A.at(i).at(k) * B.at(k).at(j);
			}
		}
	}

	return out;
}

vector<vector<float>> Blstm::outer(vector<float> a, vector<float> b)
{
	vector<vector<float>> out(a.size(), vector<float> (b.size(), 0));

	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++) {
			out.at(i).at(j) = a.at(i) * b.at(j);
		}
	}

	return out;
}

void Blstm::forward_prop(vector<vector<float>> X)
{
	_s.clear();
	_s.resize(_T, vector<float> (_hLSize, 0));
	
	_o.clear();
	_o.resize(_T, vector<float> (_oLSize, 0));

	//helper::print_matrix("U", _U);
	//helper::print_matrix("W", _W);
	//helper::print_matrix("V", _V);

	for (int t = 0; t < _T; t++)
	{
		vector<float> VdotS(_oLSize, 0);
		vector<float> UPlusW(_hLSize, 0);

		for (int iHidden = 0; iHidden < _hLSize; iHidden++)
		{
			float tmp = 0;
			for (int iIn = 0; iIn < _iLSize; iIn++)
			{
				tmp += _U.at(iHidden).at(iIn) * X.at(t).at(iIn);
			}

			float tmp2 = 0;
			for (int iHidden2 = 0; iHidden2 < _hLSize; iHidden2++)
			{
				if (t == 0)
					tmp2 += _W.at(iHidden).at(iHidden2);
				else
					tmp2 += _W.at(iHidden).at(iHidden2) * _s.at(t-1).at(iHidden2);
			}

			UPlusW.at(iHidden) = tanhyp(tmp + tmp2);
		}

		_s.at(t) = UPlusW;

		for (int iOut = 0; iOut < _oLSize; iOut++)
		{
			float tmp = 0;
			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				tmp += _V.at(iOut).at(iHidden) * _s.at(t).at(iHidden);
			}

			VdotS.at(iOut) = tmp;
		}

		_o.at(t) = softmax( VdotS );
	}

	//helper::print_matrix("_s:", _s);
}

float Blstm::calculate_loss(vector<vector<float>> Y)
{
	float L = 0;

	//helper::print_matrix("o", _o);

	for (int t = 0; t < Y.size(); t++) {
		for (int iOut = 0; iOut < Y.at(t).size(); iOut++) {
			L += -1 * Y.at(t).at(iOut) * log(_o.at(t).at(iOut));
			//L += abs(Y.at(t).at(iOut) - _o.at(t).at(iOut));
		}
	}

	L = L / _T;
	return L;
}

void Blstm::bptt(vector<vector<float>> X, vector<vector<float>> Y)
{
	//cout << "-------> bptt()" << endl;
	//helper::print_2matrices_column("_o and Y", _o, Y);

	vector<vector<float>> delta_o(_T, vector<float> (_oLSize, 0));
	
	_dU.clear();
	_dU.resize(_hLSize, vector<float> (_iLSize, 0));
	_dV.clear();
	_dV.resize(_oLSize, vector<float> (_hLSize, 0));
	_dW.clear();
	_dW.resize(_hLSize, vector<float> (_hLSize, 0));



	for (int t = 0; t < delta_o.size(); t++)
	{
		for (int iOut = 0; iOut < delta_o.at(t).size(); iOut++)
		{
			delta_o.at(t).at(iOut) = Y.at(t).at(iOut) - _o.at(t).at(iOut);
		}
	}

	//helper::print_matrix("delta_o:", delta_o);

	/*for (int t = 0; t < Y.size() ; t++)
	{
		Y.at(t) = softmax(Y.at(t));
	}*/

	for (int t = _T-1; t >= 0; t--) {
		vector<vector<float>> dVtmp = outer(delta_o.at(t), _s.at(t));
		_dV = matrix_add(_dV, dVtmp);

		vector<float> delta_t(_hLSize, 0);
		for (int iHidden = 0; iHidden < _hLSize; iHidden++)
		{
			float tmpOut = 0;
			for (int iOut = 0; iOut < _oLSize; iOut++)
			{
				tmpOut += _V.at(iOut).at(iHidden) * delta_o.at(t).at(iOut);
			}
			///	out * ( 1 - tanh^2(s(t))
			delta_t.at(iHidden) = tmpOut * (1 - ( _s.at(t).at(iHidden) * _s.at(t).at(iHidden) ));
		}

		///	bptt
		int bptt_end = t - _bpttTruncate + 1;
		if (bptt_end < 0)
			bptt_end = 0;

		for (int bptt_step = t; bptt_step >= bptt_end; bptt_step--)
		{
			vector<vector<float>> dWtmp( _hLSize, vector<float>(_hLSize, 0) );
			if (bptt_step > 0)
				dWtmp = outer(_s.at(bptt_step - 1), delta_t);
			
			_dW = matrix_add(_dW, dWtmp);

			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				for (int iIn = 0; iIn < _iLSize; iIn++)
				{
					_dU.at(iHidden).at(iIn) += delta_t.at(iHidden) * X.at(t).at(iIn);
				}
			}

			vector<float> delta_t_new(_hLSize, 0);
			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				float tmpNewDelta = 0;
				for (int iHidden2 = 0; iHidden2 < _hLSize; iHidden2++)
				{
					tmpNewDelta += _W.at(iHidden).at(iHidden2) * delta_t.at(iHidden);
				}
				if (bptt_step > 0)
				{
					delta_t.at(iHidden) = tmpNewDelta * 
						(1 - ( _s.at(bptt_step - 1).at(iHidden) * _s.at(bptt_step - 1).at(iHidden)));
				} else
					delta_t.at(iHidden) = tmpNewDelta;
			}
		}
	}
	
	float L = calculate_loss(Y);

	L *= _learningRate;

	_U = matrix_add_with_const(_U, _dU, L);
	_W = matrix_add_with_const(_W, _dW, L);
	_V = matrix_add_with_const(_V, _dV, L);

	//helper::print_matrix("_dU:", _dU);
	//helper::print_matrix("_dW:", _dW);
	//helper::print_matrix("_dV:", _dV);
}

void Blstm::random_weights()
{
	///	print function info
	//cout << "-------> random_weights()" << endl;

	///	get the random generator a seed
	srand(time(0));
	
	for (int m = 0; m < _U.size() ; m++)
	{
		for (int n = 0; n < _U.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_U.at(m).size());

			_U.at(m).at(n) = rVal;
		}
	}
	
	for (int m = 0; m < _V.size() ; m++)
	{
		for (int n = 0; n < _V.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_V.at(m).size());

			_V.at(m).at(n) = rVal;
		}
	}
	
	for (int m = 0; m < _W.size() ; m++)
	{
		for (int n = 0; n < _W.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_W.at(m).size());

			_W.at(m).at(n) = rVal;
		}
	}
	
	//helper::print_matrix("U:", _U);
	//helper::print_matrix("V:", _V);
	//helper::print_matrix("W:", _W);
}

void Blstm::print_result(vector<vector<float>> Y)
{	
	helper::print_2matrices_column("_o and Y", _o, Y);
}
