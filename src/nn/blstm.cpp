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
	/**
	 * initialize neural network parameter
	 * _T, is the length of input and output train
	 * _bpttTruncate, are the steps the bptt algorithm uses for calculation, t - bpttStep
	 * _learningRate, ist the ration the neural net is learning
	 */
	_T = T;
	_bpttTruncate = 20;
	_learningRate = lR;

	/**
	 * initliaze neural network dimension
	 * _iLSize, size of the input layer
	 * _hLSize, size of the hidden layer
	 * _oLSize, size of the output layer
	 */
	_iLSize = topo.at(0);
	_hLSize = topo.at(1);
	_oLSize = topo.at(2);

	/**
	 * create the weight matrices with the given dimensions and
	 * initialize the matrices with zeros
	 */
	_U.clear();
	_U.resize(_hLSize, vector<float> (_iLSize, 0));
	_V.clear();
	_V.resize(_oLSize, vector<float> (_hLSize, 0));
	_W.clear();
	_W.resize(_hLSize, vector<float> (_hLSize, 0));
}

float Blstm::tanhyp(float x)
{
	/// Hyperbolic activation function
	float fx = 0.0;
	fx = tanh(x);

	return fx;
}

vector<float> Blstm::softmax(vector<float> x)
{
	/// Softmax function for the output layer sigma(z)_j = exp(z_j) / sum( exp(z_k) ) , k=1..K
	/// initialze the output vector and the exponential sum with zero(s)
	vector<float> softmax(x.size(), 0);
	float xExpSum = 0;
	
	/// loop to get the exponential sum of all vector elements
	for (int i = 0; i < x.size(); i++)
	{
		xExpSum += exp(x.at(i));
	}

	/// loop every element of the input vector and save it into the outputvector
	for (int i = 0; i < x.size(); i++)
	{
		softmax.at(i) = exp(x.at(i)) / xExpSum;
	}

	return softmax;
}

vector<vector<float>> Blstm::matrix_add(vector<vector<float>> A, vector<vector<float>> B)
{
	/// initliaze output matrix with zeros, dimensions of the matrix are the same as
	/// the dimensions of the input matrices
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	/// loop every matrix element and add A_i_j and B_i_j up
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
	/// initliaze output matrix with zeros, dimensions of the matrix are the same as
	/// the dimensions of the input matrices
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	/// loop every matrix element and add A_i_j and x multiplied with B_i_j up
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
	/// check if matrices dimensions correspond to the matrix multiplication rule
	/// A with m x n and B with n x p to get C with m x p
	if (A.at(0).size() != B.size())
	{
		cout << "Matrices cant be multiplied." << endl;
	}

	/// initliaze output matrix with zeros
	vector<vector<float>> out(A.size(), vector<float> (B.at(0).size(), 0));

	/// loop every element of the output matrix
	/// every row
	for (int i = 0; i < A.size(); i++)
	{
		/// every column
		for (int j = 0; j < B.at(0).size(); j++)
		{
			/// sum up the row of A element wise multiplied by the column of B
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
	/// initialize the output matrix for the outer product of vector a and b
	/// dimension of the output matrix is m x n, with length m of vector a and length n of
	/// vector b
	vector<vector<float>> out(a.size(), vector<float> (b.size(), 0));

	/// loop every element of the output matrix out_i_j and assign the product a_i * b_j
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
	/// dump the content of the cell matrix and initialze the new cell matrix with zeros
	_s.clear();
	_s.resize(_T, vector<float> (_hLSize, 0));
	
	/// dump the content of the output matrix and initialze the new output matrix with zeros
	_o.clear();
	_o.resize(_T, vector<float> (_oLSize, 0));

	/// loop from timestep t = 0 to the end of input matrix X, X(0) to X(_T)
	for (int t = 0; t < _T; t++)
	{
		/// initialize tmp vector VdotS with zeros, this vector is a product of the
		/// multiplication of the output weight matrix _V and the cell state matrix _s at 
		/// time t and has therefore the size of the output layer
		vector<float> VdotS(_oLSize, 0);

		/// initialize tmp vector UPlusW with zeros, this vector is a product of the
		/// multiplication of the hidden weight matrix _W and the cell state matrix _s at 
		/// time t-1 added with the multiplication the input weight matrix _U and the 
		/// input matrix X at time t and has therefore the size of the hidden layer
		vector<float> UPlusW(_hLSize, 0);

		/// loop every element of the hidden layer
		for (int iHidden = 0; iHidden < _hLSize; iHidden++)
		{
			/// initialize tmp variable for input from the input layer with zero
			float tmp = 0;
			/// loop every input element
			for (int iIn = 0; iIn < _iLSize; iIn++)
			{
				/// sum up the product of every connection coming from the input layer and
				/// goes to the hidden element
				tmp += _U.at(iHidden).at(iIn) * X.at(t).at(iIn);
			}

			/// initialize tmp2 variable for the input from the recursive layer
			float tmp2 = 0;
			/// loop every recursive element
			for (int iHidden2 = 0; iHidden2 < _hLSize; iHidden2++)
			{
				/// if t = 0, set _s(t - 1) = 0
				if (t == 0)
					tmp2 += _W.at(iHidden).at(iHidden2);
				else
					tmp2 += _W.at(iHidden).at(iHidden2) * _s.at(t-1).at(iHidden2);
			}

			/// get the activated value of the hidden element
			UPlusW.at(iHidden) = tanhyp(tmp + tmp2);
		}

		/// set the cell state matrix row at time t to the calculated states
		_s.at(t) = UPlusW;


		/// loop every output element of the output layer
		for (int iOut = 0; iOut < _oLSize; iOut++)
		{
			/// initialize tmp variable for the state of the output element
			float tmp = 0;
			/// loop every hidden layer element to sum up the input connections of the
			/// output element
			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				tmp += _V.at(iOut).at(iHidden) * _s.at(t).at(iHidden);
			}

			/// allocate the calculated output value to the output vector
			VdotS.at(iOut) = tmp;
		}

		/// get the softmax vector of the output vector VdotS and save it the the output
		/// matrix _o at time t
		_o.at(t) = softmax( VdotS );
	}
}

float Blstm::calculate_loss(vector<vector<float>> Y)
{
	/// initialize the loss value with zero
	float L = 0;

	/// loop every time step in Y
	for (int t = 0; t < Y.size(); t++)
	{
		/// loop every output in Y
		for (int iOut = 0; iOut < Y.at(t).size(); iOut++)
		{
			/// sum up the loss value by calculating the product Y(t)_i * log( _o(t)_i )
			/// with i = 0 .. #output elements
			L += -1 * Y.at(t).at(iOut) * log(_o.at(t).at(iOut));
			//L += abs(Y.at(t).at(iOut) - _o.at(t).at(iOut));
		}
	}

	/// divide the loss by the length of the output train
	L = L / _T;
	return L;
}

void Blstm::bptt(vector<vector<float>> X, vector<vector<float>> Y)
{
	/// initialze delta output matrix with zeros
	/// this matrix stores the error between target and predicted output value
	vector<vector<float>> delta_o(_T, vector<float> (_oLSize, 0));
	
	/// drop the content of delta _U matrix and initialize the matrix with zeros
	/// this matrix stores the gradients which are applied to the weight matrix _U
	/// in the end of the bptt algorithm
	_dU.clear();
	_dU.resize(_hLSize, vector<float> (_iLSize, 0));
	
	/// drop the content of delta _V matrix and initialize the matrix with zeros
	/// this matrix stores the gradients which are applied to the weight matrix _V
	/// in the end of the bptt algorithm
	_dV.clear();
	_dV.resize(_oLSize, vector<float> (_hLSize, 0));
	
	/// drop the content of delta _W matrix and initialize the matrix with zeros
	/// this matrix stores the gradients which are applied to the weight matrix _W
	/// in the end of the bptt algorithm
	_dW.clear();
	_dW.resize(_hLSize, vector<float> (_hLSize, 0));


	/// loop every output value over time of the output matrix and calculate the
	/// difference (error) between target and predicted value
	for (int t = 0; t < delta_o.size(); t++)
	{
		for (int iOut = 0; iOut < delta_o.at(t).size(); iOut++)
		{
			delta_o.at(t).at(iOut) = Y.at(t).at(iOut) - _o.at(t).at(iOut);
		}
	}

	/// loop every output backwards from _T to t = 0
	for (int t = _T-1; t >= 0; t--)
	{
		/// calculate the gradient for the output weight matrix _V by multiplying the
		/// error vector delta_o at time t with every cell state in _s at time t
		vector<vector<float>> dVtmp = outer(delta_o.at(t), _s.at(t));
		
		/// add the calculatet matrix up with the previous calculated matrix
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

	_U = matrix_add_with_const(_U, _dU, 0.001);
	_W = matrix_add_with_const(_W, _dW, 0.001);
	_V = matrix_add_with_const(_V, _dV, 0.001);

	//helper::print_matrix("_dU:", _dU);
	//helper::print_matrix("_dW:", _dW);
	//helper::print_matrix("_dV:", _dV);
}

void Blstm::random_weights()
{
	///	give the random generator a seed
	srand(time(0));
	
	/// loop every element in matrix _U and assign a random value
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
	
	/// loop every element in matrix _V and assign a random value
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
	
	/// loop every element in matrix _W and assign a random value
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
}

void Blstm::print_result(vector<vector<float>> Y)
{	
	/// print predicted value matrix _o next to target value matrix Y
	helper::print_2matrices_column("_o and Y", _o, Y);

	/// print weight matrix _U, _V and _W
	helper::print_matrix("_U", _U);
	helper::print_matrix("_W", _W);
	helper::print_matrix("_V", _V);
}
