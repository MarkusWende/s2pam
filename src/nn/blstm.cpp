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
	
	_Wi.clear();
	_Wi.resize(_hLSize + _iLSize, vector<float> (_hLSize, 0));
	
	_Wf.clear();
	_Wf.resize(_hLSize + _iLSize, vector<float> (_hLSize, 0));
	
	_Wo.clear();
	_Wo.resize(_hLSize + _iLSize, vector<float> (_hLSize, 0));
	
	_Wc.clear();
	_Wc.resize(_hLSize + _iLSize, vector<float> (_hLSize, 0));
	
	_Wy.clear();
	_Wy.resize(_hLSize, vector<float> (_oLSize, 0));
}

float Blstm::tanhyp(float x)
{
	/// Hyperbolic activation function
	float fx = 0.0;
	fx = tanh(x);

	return fx;
}

vector<float> Blstm::tanhyp(vector<float> x)
{
	/// initialze the output vector
	vector<float> tan(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		tan.at(i) = tanhyp(x.at(i));
	}

	return tan;
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

	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		softmax.at(i) = exp(x.at(i)) / xExpSum;
	}

	return softmax;
}

float Blstm::sigmoid(float x)
{
	/// initialze the output
	float sig = 0;
	
	sig = 1 / (1 + exp(-x));

	return sig;
}

vector<float> Blstm::sigmoid(vector<float> x)
{
	/// initialze the output vector
	vector<float> sig(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		sig.at(i) = sigmoid(x.at(i));
	}

	return sig;
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

vector<float> Blstm::vec_matrix_mult(vector<float> a, vector<vector<float>> B)
{
	/// check if matrices dimensions correspond to the matrix multiplication rule
	/// A with m x n and B with n x p to get C with m x p
	if (a.size() != B.size())
	{
		cout << "Vector and matrix cant be multiplied." << endl;
	}

	/// initliaze output vector with zeros
	vector<float> out(B.at(0).size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		/// sum up the elements in a multiplied by the elements in the column of B
		for (int j = 0; j < a.size(); j++)
		{
			out.at(i) = a.at(j) * B.at(j).at(i);
		}
	}

	return out;
}

vector<float> Blstm::vec_ele_add(vector<float> a, vector<float> b)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be added element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<float> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) + b.at(i);
	}

	return out;
}

vector<float> Blstm::vec_ele_mult(vector<float> a, vector<float> b)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be multiplied element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<float> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) * b.at(i);
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

vector<float> Blstm::vec_concat(vector<float> a, vector<float> b)
{
	vector<float> out(a.size() + b.size(), 0);
	int index = 0;
	
	for (int i = 0; i < a.size(); i++)
	{
		out.at(i) = a.at(i);
		index = i;
	}

	for (int j = 0; j < b.size(); j++)
	{
		out.at(index + j) = b.at(j);
	}

	return out;
}

void Blstm::forward_prop(vector<vector<float>> X)
{
	/// dump the content of the cell output matrix and initialze the
	/// new cell output matrix with zeros
	_s.clear();
	_s.resize(_T, vector<float> (_hLSize, 0));
	_h.clear();
	_h.resize(_T, vector<float> (_hLSize, 0));
	
	/// dump the content of the cell matrix and initialze the new cell matrix with zeros
	_c.clear();
	_c.resize(_T, vector<float> (_hLSize, 0));
	
	/// dump the content of the output matrix and initialze the new output matrix with zeros
	_o.clear();
	_o.resize(_T, vector<float> (_oLSize, 0));
	_y.clear();
	_y.resize(_T, vector<float> (_oLSize, 0));

	vector<float> h_old(_hLSize, 0);

	/// loop from timestep t = 0 to the end of input matrix X, X(0) to X(_T)
	for (int t = 0; t < _T; t++)
	{
		/// initialize tmp vector VdotS with zeros, this vector is a product of the
		/// multiplication of the output weight matrix _V and the cell state matrix _s at 
		/// time t and has therefore the size of the output layer
		vector<float> VdotS(_oLSize, 0);

		/**
		 *														    ^
		 *													   h(t)	|	
		 *				=====================================================
		 *				#											|		#
		 *	C(t-1) ---->#------	x ---------	+ --------------┬---------------#----> C(t)
		 *				#	hf	|	hi		|			 |tanh|		|		#
		 *				#		|	┌------>x		ho		|		|		#
		 *				#		|	|		| hc	┌------>x		|		#
		 *				#	   |σ| |σ|   |tanh|	   |σ|		|		|		#
		 *	h(t-1) ---->#---┬---┴---┴-------┴-------┴		└-------┴-------#----> h(t)
		 *				#	|												#
		 *				=====================================================
		 *					^
		 *					| x(t)
		 *
		 */
		vector<float> hf(_hLSize, 0);
		vector<float> hi(_hLSize, 0);
		vector<float> ho(_hLSize, 0);
		vector<float> hc(_hLSize, 0);

		vector<float> x_t(_iLSize, 0);
		x_t = X.at(t);
		vector<float> cell_in(_iLSize + _hLSize, 0);

		vector<float> h_old(_hLSize, 0);
		if (t > 0)
			h_old = _h.at(t-1);

		cell_in = vec_concat(h_old, x_t);

		hf = vec_matrix_mult(cell_in, _Wf);
		hf = sigmoid(hf);

		hi = vec_matrix_mult(cell_in, _Wi);
		hi = sigmoid(hi);

		ho = vec_matrix_mult(cell_in, _Wo);
		ho = sigmoid(ho);
		
		hc = vec_matrix_mult(cell_in, _Wc);
		hc = tanhyp(hc);

		vector<float> c_old(_hLSize, 0);
		if (t > 0)
			c_old = _c.at(t-1);
		
		_c.at(t) = vec_ele_add( vec_ele_mult(hf, c_old), vec_ele_mult(hi, hc) );
		_h.at(t) = vec_ele_mult( ho, tanhyp(_c.at(t)) );
		_y.at(t) = softmax( vec_matrix_mult(_h.at(t), _Wy) );
		
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
			_s.at(t).at(iHidden) = tanhyp(tmp + tmp2);
		}

		/// set the cell state matrix row at time t to the calculated states


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
			L += -1 * Y.at(t).at(iOut) * log(_y.at(t).at(iOut));
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
	
	/// loop every element in matrix _Wf and assign a random value
	for (int m = 0; m < _Wf.size() ; m++)
	{
		for (int n = 0; n < _Wf.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wf.at(m).size());

			_Wf.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wi and assign a random value
	for (int m = 0; m < _Wi.size() ; m++)
	{
		for (int n = 0; n < _Wi.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wi.at(m).size());

			_Wi.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wc and assign a random value
	for (int m = 0; m < _Wc.size() ; m++)
	{
		for (int n = 0; n < _Wc.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wc.at(m).size());

			_Wc.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wo and assign a random value
	for (int m = 0; m < _Wo.size() ; m++)
	{
		for (int n = 0; n < _Wo.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wo.at(m).size());

			_Wo.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wy and assign a random value
	for (int m = 0; m < _Wy.size() ; m++)
	{
		for (int n = 0; n < _Wy.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wy.at(m).size());

			_Wy.at(m).at(n) = rVal;
		}
	}
}

void Blstm::print_result(vector<vector<float>> Y)
{	
	/// print predicted value matrix _o next to target value matrix Y
	//helper::print_2matrices_column("_o and Y", _o, Y);
	helper::print_2matrices_column("_y and Y", _y, Y);

	/// print weight matrix _U, _V and _W
	//helper::print_matrix("_U", _U);
	//helper::print_matrix("_W", _W);
	//helper::print_matrix("_V", _V);
	helper::print_matrix("_Wf", _Wf);
	helper::print_matrix("_Wi", _Wi);
	helper::print_matrix("_Wc", _Wc);
	helper::print_matrix("_Wo", _Wo);
	helper::print_matrix("_Wy", _Wy);
}

void Blstm::render_weights(int index)
{	
	/// render weight matrix _U
	vector<float> _u;
	unsigned int heightU, widthU;
	vector<vector<float>> ampU(_U.size(), vector<float>(_U.at(0).size(), 0));
	ampU = matrix_add_with_const(ampU, _U, 10);

	helper::matrix_to_vector(ampU, heightU, widthU, _u);
	render::vector_to_PNG("weight_U_", std::to_string(index), "exp", heightU, widthU, _u);


	/// render weight matrix _W
	vector<float> _w;
	unsigned int heightW, widthW;
	vector<vector<float>> ampW(_W.size(), vector<float>(_W.at(0).size(), 0));
	ampW = matrix_add_with_const(ampW, _W, 10);

	helper::matrix_to_vector(ampW, heightW, widthW, _w);
	render::vector_to_PNG("weight_W_", std::to_string(index), "exp", heightW, widthW, _w);


	/// render weight matrix _V
	vector<float> _v;
	unsigned int heightV, widthV;
	vector<vector<float>> ampV(_V.size(), vector<float>(_V.at(0).size(), 0));
	ampV = matrix_add_with_const(ampV, _V, 10);

	helper::matrix_to_vector(ampV, heightV, widthV, _v);
	render::vector_to_PNG("weight_V_", std::to_string(index), "exp", heightV, widthV, _v);
}

void Blstm::save()
{
	/// file name of the neural network binary
	string filename = "data.bin";

	///	append the parent folder the file is stored in
	filename.insert(0,"./data/nn/");

	///	construct ofstream object and initialze filename
	ofstream outputFile;
	outputFile.open(filename);

	/// write topology of the nn to the file
	outputFile << "topology " << _iLSize << ' ' << _hLSize << ' ' << _oLSize << endl;

	/// write train length to the file
	outputFile << "T " << _T << endl;

	/// write bptt truncate to the file
	outputFile << "bpttTrunc " << _bpttTruncate << endl;

	/// write learning rate to the file
	outputFile << "lR " << _learningRate << endl;

	/**
	 * write weight matrix _U, _W and _V to the file
	 */

	/// _U
	outputFile << "U" << endl;
	/// get matrix dimensions
	int height = _U.size();
	int width = _U.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _U.at(i).at(j) << ' ';
			else
				outputFile << _U.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _W
	outputFile << "W" << endl;
	/// get matrix dimensions
	height = _W.size();
	width = _W.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _W.at(i).at(j) << ' ';
			else
				outputFile << _W.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _V
	outputFile << "V" << endl;
	/// get matrix dimensions
	height = _V.size();
	width = _V.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _V.at(i).at(j) << ' ';
			else
				outputFile << _V.at(i).at(j);
		}
		outputFile << endl;
	}

	///	close file
	outputFile.close();
}

void Blstm::load()
{

}
