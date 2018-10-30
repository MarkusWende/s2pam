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
using namespace helper;

Blstm::Blstm(vector<unsigned> topo, int T, double lR)
{
	/**
	 * initialize neural network parameter
	 * _T, is the length of input and output train
	 * _bpttTruncate, are the steps the bptt algorithm uses for calculation, t - bpttStep
	 * _learningRate, ist the ration the neural net is learning
	 */
	_T = T;
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
	_Wi.clear();
	_Wi.resize(_iLSize, vector<double> (_hLSize, 0));
	
	_Wf.clear();
	_Wf.resize(_iLSize, vector<double> (_hLSize, 0));
	
	_Wo.clear();
	_Wo.resize(_iLSize, vector<double> (_hLSize, 0));
	
	_Wz.clear();
	_Wz.resize(_iLSize, vector<double> (_hLSize, 0));
	
	_Wy.clear();
	_Wy.resize(_hLSize, vector<double> (_oLSize, 0));
	
	
	_Ri.clear();
	_Ri.resize(_hLSize, vector<double> (_hLSize, 0));
	
	_Rf.clear();
	_Rf.resize(_hLSize, vector<double> (_hLSize, 0));
	
	_Ro.clear();
	_Ro.resize(_hLSize, vector<double> (_hLSize, 0));
	
	_Rz.clear();
	_Rz.resize(_hLSize, vector<double> (_hLSize, 0));
}

double Blstm::tanhyp(double x)
{
	/// Hyperbolic activation function
	double fx = 0.0;
	fx = tanh(x);

	return fx;
}

double Blstm::dtanhyp(double x)
{
	/// Hyperbolic activation function
	double dfx = 0.0;
	dfx = 1 - (tanh(x) * tanh(x));

	return dfx;
}

vector<double> Blstm::tanhyp(vector<double> x)
{
	/// initialze the output vector
	vector<double> tan(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		tan.at(i) = tanhyp(x.at(i));
	}

	return tan;
}

vector<double> Blstm::dtanhyp(vector<double> x)
{
	/// initialze the output vector
	vector<double> dtan(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		dtan.at(i) = dtanhyp(x.at(i));
	}

	return dtan;
}

vector<double> Blstm::softmax(vector<double> x)
{
	/// Softmax function for the output layer sigma(z)_j = exp(z_j) / sum( exp(z_k) ) , k=1..K
	/// initialze the output vector and the exponential sum with zero(s)
	vector<double> softmax(x.size(), 0);
	double xExpSum = 0;
	
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

double Blstm::sigmoid(double x)
{
	/// initialze the output
	double sig = 0;
	
	sig = 1 / (1 + exp(-x));

	return sig;
}

double Blstm::dsigmoid(double x)
{
	/// initialze the output
	double dsig = 0;
	
	dsig = exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));

	return dsig;
}

vector<double> Blstm::sigmoid(vector<double> x)
{
	/// initialze the output vector
	vector<double> sig(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		sig.at(i) = sigmoid(x.at(i));
	}

	return sig;
}

vector<double> Blstm::dsigmoid(vector<double> x)
{
	/// initialze the output vector
	vector<double> dsig(x.size(), 0);
	
	/// loop every element of the input vector and save it into the output vector
	for (int i = 0; i < x.size(); i++)
	{
		dsig.at(i) = dsigmoid(x.at(i));
	}

	return dsig;
}

void Blstm::forward_prop(vector<vector<double>> X)
{
	/// dump the content of the cell output matrix and initialze the
	/// new cell output matrix with zeros
	_y.clear();
	_y.resize(_T, vector<double> (_hLSize, 0));
	_f.clear();
	_f.resize(_T, vector<double> (_hLSize, 0));
	_i.clear();
	_i.resize(_T, vector<double> (_hLSize, 0));
	_o.clear();
	_o.resize(_T, vector<double> (_hLSize, 0));
	_z.clear();
	_z.resize(_T, vector<double> (_hLSize, 0));
	
	_f_head.clear();
	_f_head.resize(_T, vector<double> (_hLSize, 0));
	_i_head.clear();
	_i_head.resize(_T, vector<double> (_hLSize, 0));
	_o_head.clear();
	_o_head.resize(_T, vector<double> (_hLSize, 0));
	_z_head.clear();
	_z_head.resize(_T, vector<double> (_hLSize, 0));
	
	/// dump the content of the cell matrix and initialze the new cell matrix with zeros
	_c.clear();
	_c.resize(_T, vector<double> (_hLSize, 0));
	
	/// dump the content of the output matrix and initialze the new output matrix with zeros
	_prediction.clear();
	_prediction.resize(_T, vector<double> (_oLSize, 0));

	vector<double> y_tMinus1(_hLSize, 0);

	/// loop from timestep t = 0 to the end of input matrix X, X(0) to X(_T)
	for (int t = 0; t < _T; t++)
	{
		/**
		 *														    ^
		 *													   y(t)	|	
		 *				=====================================================
		 *				#							c				|		#
		 *	c(t-1) ---->#------	x ---------	+ --------------┬---------------#----> c(t)
		 *				#	  f	|		i	|			 |tanh|		|		#
		 *				#		|	┌------>x			o	|		|		#
		 *				#		|	|		| z		┌------>x		|		#
		 *				#	   |σ| |σ|   |tanh|	   |σ|		|		|		#
		 *	y(t-1) ---->#---┬---┴---┴-------┴-------┴		└-------┴-------#----> y(t)
		 *				#	|												#
		 *				=====================================================
		 *					^
		 *					| x(t)
		 *
		 */
		vector<double> f(_hLSize, 0);
		vector<double> i(_hLSize, 0);
		vector<double> o(_hLSize, 0);
		vector<double> z(_hLSize, 0);
		
		vector<double> f_head(_hLSize, 0);
		vector<double> i_head(_hLSize, 0);
		vector<double> o_head(_hLSize, 0);
		vector<double> z_head(_hLSize, 0);
		
		vector<double> c(_hLSize, 0);

		vector<double> x_t(_iLSize, 0);
		x_t = X.at(t);

		f_head = vec_matrix_mult(x_t, _Wf);
		f_head = vec_ele_add( f_head, vec_matrix_mult(y_tMinus1, _Rf) );
		_f_head.at(t) = f_head;
		f = sigmoid(f_head);
		_f.at(t) = f;

		i_head = vec_matrix_mult(x_t, _Wi);
		i_head = vec_ele_add( i_head, vec_matrix_mult(y_tMinus1, _Ri) );
		_i_head.at(t) = i_head;
		i = sigmoid(i_head);
		_i.at(t) = i;

		o_head = vec_matrix_mult(x_t, _Wo);
		o_head = vec_ele_add( o_head, vec_matrix_mult(y_tMinus1, _Ro) );
		_o_head.at(t) = o_head;
		o = sigmoid(o_head);
		_o.at(t) = o;
		
		z_head = vec_matrix_mult(x_t, _Wz);
		z_head = vec_ele_add( z_head, vec_matrix_mult(y_tMinus1, _Rz) );
		_z_head.at(t) = z_head;
		z = tanhyp(z_head);
		_z.at(t) = z;
		

		vector<double> c_old(_hLSize, 0);
		if (t > 0)
			c_old = _c.at(t-1);
		
		c = vec_ele_mult(z, i);
		c = vec_ele_add( c, vec_ele_mult(f, c_old) );
		_c.at(t) = c;

		_y.at(t) = vec_ele_mult( o, tanhyp(c) );

		_prediction.at(t) = softmax( vec_matrix_mult(_y.at(t), _Wy) );
	}
}

double Blstm::calculate_loss(vector<vector<double>> Y)
{
	/// initialize the loss value with zero
	double L = 0;

	/// loop every time step in Y
	for (int t = 0; t < Y.size(); t++)
	{
		/// loop every output in Y
		for (int iOut = 0; iOut < Y.at(t).size(); iOut++)
		{
			/// sum up the loss value by calculating the product Y(t)_i * log( _o(t)_i )
			/// with i = 0 .. output layer size
			//L += Y.at(t).at(iOut) * log(_prediction.at(t).at(iOut));
			L += abs(Y.at(t).at(iOut) - _prediction.at(t).at(iOut));
		}
	}

	/// divide the loss by the length of the output train
	L = L / _T;
	//cout << "\tL: " << L;
	return L;
}

void Blstm::bptt(vector<vector<double>> X, vector<vector<double>> Y)
{
	vector<vector<double>> dWy(_hLSize, vector<double> (_oLSize, 0));
	
	vector<vector<double>> dWi(_iLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dWf(_iLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dWo(_iLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dWz(_iLSize, vector<double> (_hLSize, 0));

	vector<vector<double>> dRi(_hLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dRf(_hLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dRo(_hLSize, vector<double> (_hLSize, 0));
	vector<vector<double>> dRz(_hLSize, vector<double> (_hLSize, 0));

	vector<double> dy_next(_hLSize, 0);
	vector<double> dc_next(_hLSize, 0);
	
	vector<double> df_head(_hLSize, 0);
	vector<double> di_head(_hLSize, 0);
	vector<double> do_head(_hLSize, 0);
	vector<double> dz_head(_hLSize, 0);

	vector<double> x_t(_iLSize, 0);
	vector<double> y_t(_iLSize, 0);

	/// loop every output backwards from _T-1 to t = 0
	for (int t = _T-1; t >= 0; t--)
	{
		y_t = _y.at(t);
		x_t = X.at(t);

		/// dRf, dRi, dRo and dRz
		dRf = matrix_add(dRf, outer(y_t, df_head));
		dRi = matrix_add(dRi, outer(y_t, di_head));
		dRo = matrix_add(dRo, outer(y_t, do_head));
		dRz = matrix_add(dRz, outer(y_t, dz_head));

		/// Error dE/dy(t)
		/// dE/dy(t) = prediction(t) - y(t)
		vector<double> deltaT(_oLSize, 0);
		deltaT = vec_ele_sub(Y.at(t), _prediction.at(t));
		
		/// dy(t) = delta(t) + dy(t+1)
		vector<double> dy(_hLSize, 0);
		dy = vec_matrix_mult(deltaT, matrix_T(_Wy));
		dy = vec_ele_add(dy, dy_next);

		/// do_head(t) = dy(t) * tanh( c(t) ) * dsig( o_head(t) )
		do_head = vec_ele_mult(tanhyp(_c.at(t)), dy);
		do_head = vec_ele_mult(dsigmoid(_o_head.at(t)), do_head);

		/// dc(t) = dy(t) * o(t) * dtanh( c(t) ) + dc(t+1) * f(t+1)
		vector<double> dc = vec_ele_mult(_o.at(t), dy);
		dc = vec_ele_mult(dc, dtanhyp(_c.at(t)));
		dc = vec_ele_add(dc, dc_next);
		
		/// dc(t+1) * f(t+1)
		dc_next = vec_ele_mult(_f.at(t), dc);

		/// df_head(t) = dc(t) * c(t-1) * dsig( f_head(t) )
		if (t > 0)
			df_head = vec_ele_mult(_c.at(t-1), dc);
		df_head = vec_ele_mult(dsigmoid(_f_head.at(t)), df_head);	
		
		/// di_head(t) = dc(t) * z(t) * dsig( i_head(t) )
		di_head = vec_ele_mult(_z.at(t), dc);
		di_head = vec_ele_mult(dsigmoid(_i_head.at(t)), di_head);

		/// dz_head(t) = dc(t) * i(t) * dtanh( _z_head(t) )
		dz_head = vec_ele_mult(_i.at(t), dc);
		dz_head = vec_ele_mult(dtanhyp(_z_head.at(t)), dz_head);

		/// calc gradients
		
		/// dWy, gradient of _Wy
		dWy = matrix_add(dWy, outer(y_t, deltaT) );
		
		/// dWf, dWi, dWo and dWz
		dWf = matrix_add(dWf, outer(x_t, df_head));
		dWi = matrix_add(dWi, outer(x_t, di_head));
		dWo = matrix_add(dWo, outer(x_t, do_head));
		dWz = matrix_add(dWz, outer(x_t, dz_head));

		/// dy_next
		vector<double> dy_Rf = vec_matrix_mult(df_head, matrix_T(_Rf));
		vector<double> dy_Ri = vec_matrix_mult(di_head, matrix_T(_Ri));
		vector<double> dy_Ro = vec_matrix_mult(do_head, matrix_T(_Ro));
		vector<double> dy_Rz = vec_matrix_mult(dz_head, matrix_T(_Rz));

		vector<double> dy_R = vec_ele_add(dy_Ro, dy_Rz);
		dy_R = vec_ele_add(dy_R, dy_Ri);
		dy_R = vec_ele_add(dy_R, dy_Rf);
		dy_next = dy_R;
	}
	
	double L = calculate_loss(Y);

	L = L * _learningRate;
	
	_Wy = matrix_add_with_const(_Wy, dWy, L);
	_Wf = matrix_add_with_const(_Wf, dWf, L);
	_Wi = matrix_add_with_const(_Wi, dWi, L);
	_Wo = matrix_add_with_const(_Wo, dWo, L);
	_Wz = matrix_add_with_const(_Wz, dWz, L);

	_Rf = matrix_add_with_const(_Rf, dRf, L);
	_Ri = matrix_add_with_const(_Ri, dRi, L);
	_Ro = matrix_add_with_const(_Ro, dRo, L);
	_Rz = matrix_add_with_const(_Rz, dRz, L);
}

void Blstm::random_weights()
{
	///	give the random generator a seed
	srand(time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);

	/// normal distribution, mean = 0.0, deviation = 0.1
	std::normal_distribution<double> distribution (0.0,0.1);
	
	/// loop every element in matrix _Wf and assign a random value
	for (int m = 0; m < _Wf.size() ; m++)
	{
		for (int n = 0; n < _Wf.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
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
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wi.at(m).size());

			_Wi.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wz and assign a random value
	for (int m = 0; m < _Wz.size() ; m++)
	{
		for (int n = 0; n < _Wz.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wz.at(m).size());

			_Wz.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Wo and assign a random value
	for (int m = 0; m < _Wo.size() ; m++)
	{
		for (int n = 0; n < _Wo.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
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
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Wy.at(m).size());

			_Wy.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Rf and assign a random value
	for (int m = 0; m < _Rf.size() ; m++)
	{
		for (int n = 0; n < _Rf.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Rf.at(m).size());

			_Rf.at(m).at(n) = rVal;
		}
	}
	
	/// loop every element in matrix _Ri and assign a random value
	for (int m = 0; m < _Ri.size() ; m++)
	{
		for (int n = 0; n < _Ri.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Ri.at(m).size());

			_Ri.at(m).at(n) = rVal;
		}
	}

	/// loop every element in matrix _Rz and assign a random value
	for (int m = 0; m < _Rz.size() ; m++)
	{
		for (int n = 0; n < _Rz.at(m).size() ; n++)
		{
			_Rz.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Ro and assign a random value
	for (int m = 0; m < _Ro.size() ; m++)
	{
		for (int n = 0; n < _Ro.at(m).size() ; n++)
		{
			double rVal = rand() / double(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_Ro.at(m).size());

			_Ro.at(m).at(n) = rVal;
		}
	}
}

void Blstm::print_result(vector<vector<double>> Y)
{	
	/// print predicted value matrix _o next to target value matrix Y
	//helper::print_2matrices_column("_o and Y", _o, Y);
	
	print_2matrices_column("_prediction and Y", _prediction, Y);

	/// print weight matrix _U, _V and _W
	//helper::print_matrix("_U", _U);
	//helper::print_matrix("_W", _W);
	//helper::print_matrix("_V", _V);
	//helper::print_matrix("_Wf", _Wf);
	//helper::print_matrix("_Wi", _Wi);
	//helper::print_matrix("_Wc", _Wc);
	//helper::print_matrix("_Wo", _Wo);
	print_matrix("_Rz", _Rz);
}

void Blstm::render_weights(int index)
{	
	/// render weight matrix _Rz
	vector<double> rz;
	unsigned int heightRz, widthRz;
	vector<vector<double>> ampRz(_Rz.size(), vector<double>(_Rz.at(0).size(), 0));
	ampRz = matrix_add_with_const(ampRz, _Rz, 1);

	matrix_to_vector(ampRz, heightRz, widthRz, rz);
	render::vector_to_PNG("_Rz", std::to_string(index), "exp", heightRz, widthRz, rz);
}

bool Blstm::check_weight_sum()
{
	cout << "\tSUM Rz: " << matrix_sum(_Rz) << endl;

	return true;
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

	/// write learning rate to the file
	outputFile << "lR " << _learningRate << endl;

	/**
	 * write weight matrix _U, _W and _V to the file
	 */

	/// _Wy
	outputFile << "Wy" << endl;
	/// get matrix dimensions
	int height = _Wy.size();
	int width = _Wy.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Wy.at(i).at(j) << ' ';
			else
				outputFile << _Wy.at(i).at(j);
		}
		outputFile << endl;
	}

	///	close file
	outputFile.close();
}

void Blstm::load()
{

}
