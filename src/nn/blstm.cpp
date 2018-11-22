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

Blstm::Blstm(string filename)
{
	///	construct ifstream object and initialze filename
	ifstream inputFile;
	inputFile.open(filename, std::ifstream::in);
	
	bool head = true;
	while (head)
	{
		///	safe input line from file as a string and initialize line counter
		string line;
		string label;

		getline(inputFile, line);
		
		istringstream ss(line);
		ss >> label;
		
		if (label.compare("topology") == 0)
		{
			int val;
			ss >> val;
			_iLSize = val;
			ss >> val;
			_hLSize = val;
			ss >> val;
			_oLSize = val;
		} else if (label.compare("T") == 0)
		{
			int val;
			ss >> val;
			_T = val;
		} else if (label.compare("lR") == 0)
		{
			double val;
			ss >> val;
			_learningRate = val;
			head = false;
		} else
		{
			cout << "No Header! Exit(0)" << endl;
			exit(0);
		}
	}

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

	_bi.clear();
	_bi.resize(_hLSize, 0);
	_bf.clear();
	_bf.resize(_hLSize, 0);
	_bo.clear();
	_bo.resize(_hLSize, 0);
	_bz.clear();
	_bz.resize(_hLSize, 0);

	_pi.clear();
	_pi.resize(_hLSize, 0);
	_pf;
	_pf.resize(_hLSize, 0);
	_po;
	_po.resize(_hLSize, 0);

	/**
	 * create the weight matrices with the given dimensions and
	 * initialize the matrices with zeros
	 */
	_b_Wi.clear();
	_b_Wi.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wf.clear();
	_b_Wf.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wo.clear();
	_b_Wo.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wz.clear();
	_b_Wz.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wy.clear();
	_b_Wy.resize(_hLSize, vector<double> (_oLSize, 0));
	
	_b_Ri.clear();
	_b_Ri.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Rf.clear();
	_b_Rf.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Ro.clear();
	_b_Ro.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Rz.clear();
	_b_Rz.resize(_hLSize, vector<double> (_hLSize, 0));

	_b_bi.clear();
	_b_bi.resize(_hLSize, 0);
	_b_bf.clear();
	_b_bf.resize(_hLSize, 0);
	_b_bo.clear();
	_b_bo.resize(_hLSize, 0);
	_b_bz.clear();
	_b_bz.resize(_hLSize, 0);

	_b_pi.clear();
	_b_pi.resize(_hLSize, 0);
	_b_pf;
	_b_pf.resize(_hLSize, 0);
	_b_po;
	_b_po.resize(_hLSize, 0);
	
	int counter = 0;
	bool data = true;

	string line;
	string label;
	istringstream ss(line);

	///	loop over the lines in a file
	while (data)
	{
		string line;
		string label;
		
		getline(inputFile, line);
		
		istringstream ss(line);
		ss >> label;

		bool done = false;
		int row = 0;
		double val = 0;

		///	initialize new matrix row
		//mMfccCoeffs.push_back(vector<double> (0,0));

		if (label.compare("Wy") == 0)
		{
			while (row < _Wy.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Wy.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bWy") == 0)
		{
			while (row < _b_Wy.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Wy.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Wi") == 0)
		{
			while (row < _Wi.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Wi.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bWi") == 0)
		{
			while (row < _b_Wi.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Wi.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Wf") == 0)
		{
			while (row < _Wf.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Wf.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bWf") == 0)
		{
			while (row < _b_Wf.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Wf.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Wo") == 0)
		{
			while (row < _Wo.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Wo.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bWo") == 0)
		{
			while (row < _b_Wo.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Wo.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Wz") == 0)
		{
			while (row < _Wz.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Wz.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bWz") == 0)
		{
			while (row < _b_Wz.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Wz.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Ri") == 0)
		{
			while (row < _Ri.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Ri.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bRi") == 0)
		{
			while (row < _b_Ri.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Ri.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Rf") == 0)
		{
			while (row < _Rf.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Rf.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bRf") == 0)
		{
			while (row < _b_Rf.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Rf.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Ro") == 0)
		{
			while (row < _Ro.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Ro.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bRo") == 0)
		{
			while (row < _b_Ro.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Ro.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("Rz") == 0)
		{
			while (row < _Rz.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_Rz.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bRz") == 0)
		{
			while (row < _b_Rz.size())
			{
				string dataLine;
				int element = 0;
				getline(inputFile, dataLine);

				istringstream dataSS(dataLine);
				
				dataSS.str( dataLine );
				while (dataSS >> val)
				{
					_b_Rz.at(row).at(element) = val;
					element++;
				}
				row++;
			}
		} else if (label.compare("bi") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_bi.at(element) = val;
				element++;
			}
		} else if (label.compare("bbi") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_bi.at(element) = val;
				element++;
			}
		} else if (label.compare("bf") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_bf.at(element) = val;
				element++;
			}
		} else if (label.compare("bbf") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_bf.at(element) = val;
				element++;
			}
		} else if (label.compare("bo") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_bo.at(element) = val;
				element++;
			}
		} else if (label.compare("bbo") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_bo.at(element) = val;
				element++;
			}
		} else if (label.compare("bz") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_bz.at(element) = val;
				element++;
			}
		} else if (label.compare("bbz") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_bz.at(element) = val;
				element++;
			}
		} else if (label.compare("pi") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_pi.at(element) = val;
				element++;
			}
		} else if (label.compare("bpi") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_pi.at(element) = val;
				element++;
			}
		} else if (label.compare("pf") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_pf.at(element) = val;
				element++;
			}
		} else if (label.compare("bpf") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_pf.at(element) = val;
				element++;
			}
		} else if (label.compare("po") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_po.at(element) = val;
				element++;
			}
		} else if (label.compare("bpo") == 0)
		{
			string dataLine;
			int element = 0;
			getline(inputFile, dataLine);

			istringstream dataSS(dataLine);
			
			dataSS.str( dataLine );
			while (dataSS >> val)
			{
				_b_po.at(element) = val;
				element++;
			}
		} else
			data = false;
	}

	///	close file
	inputFile.close();
}

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

	_bi.clear();
	_bi.resize(_hLSize, 0);
	_bf.clear();
	_bf.resize(_hLSize, 0);
	_bo.clear();
	_bo.resize(_hLSize, 0);
	_bz.clear();
	_bz.resize(_hLSize, 0);

	_pi.clear();
	_pi.resize(_hLSize, 0);
	_pf;
	_pf.resize(_hLSize, 0);
	_po;
	_po.resize(_hLSize, 0);

	/**
	 * create the weight matrices with the given dimensions and
	 * initialize the matrices with zeros
	 */
	_b_Wi.clear();
	_b_Wi.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wf.clear();
	_b_Wf.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wo.clear();
	_b_Wo.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wz.clear();
	_b_Wz.resize(_iLSize, vector<double> (_hLSize, 0));
	_b_Wy.clear();
	_b_Wy.resize(_hLSize, vector<double> (_oLSize, 0));
	
	_b_Ri.clear();
	_b_Ri.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Rf.clear();
	_b_Rf.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Ro.clear();
	_b_Ro.resize(_hLSize, vector<double> (_hLSize, 0));
	_b_Rz.clear();
	_b_Rz.resize(_hLSize, vector<double> (_hLSize, 0));

	_b_bi.clear();
	_b_bi.resize(_hLSize, 0);
	_b_bf.clear();
	_b_bf.resize(_hLSize, 0);
	_b_bo.clear();
	_b_bo.resize(_hLSize, 0);
	_b_bz.clear();
	_b_bz.resize(_hLSize, 0);

	_b_pi.clear();
	_b_pi.resize(_hLSize, 0);
	_b_pf;
	_b_pf.resize(_hLSize, 0);
	_b_po;
	_b_po.resize(_hLSize, 0);
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
		if (isnan(softmax.at(i)))
		{
			cerr << "x.at(i): " << x.at(i) << "\t xExpSum: " << xExpSum << endl;
			exit(0);
		}
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

void Blstm::feed_forward(vector<vector<double>> X)
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
	
	vector<double> y_tMinus1(_hLSize, 0);
	vector<double> c_old(_hLSize, 0);

	/// loop from timestep t = 0 to the end of input matrix X, X(0) to X(_T)
	for (int t = 0; t < _T; t++)
	{
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
		f_head = vec_ele_add( f_head, vec_ele_mult(_pf, c_old) );
		f_head = vec_ele_add( f_head, _bf);
		_f_head.at(t) = f_head;
		f = sigmoid(f_head);
		_f.at(t) = f;

		i_head = vec_matrix_mult(x_t, _Wi);
		i_head = vec_ele_add( i_head, vec_matrix_mult(y_tMinus1, _Ri) );
		i_head = vec_ele_add( i_head, vec_ele_mult(_pi, c_old) );
		i_head = vec_ele_add( i_head, _bi);
		_i_head.at(t) = i_head;
		i = sigmoid(i_head);
		_i.at(t) = i;

		o_head = vec_matrix_mult(x_t, _Wo);
		o_head = vec_ele_add( o_head, vec_matrix_mult(y_tMinus1, _Ro) );
		o_head = vec_ele_add( o_head, vec_ele_mult(_po, c_old) );
		o_head = vec_ele_add( o_head, _bo);
		_o_head.at(t) = o_head;
		o = sigmoid(o_head);
		_o.at(t) = o;
		
		z_head = vec_matrix_mult(x_t, _Wz);
		z_head = vec_ele_add( z_head, vec_matrix_mult(y_tMinus1, _Rz) );
		z_head = vec_ele_add( z_head, _bz);
		_z_head.at(t) = z_head;
		z = tanhyp(z_head);
		_z.at(t) = z;

		c = vec_ele_mult(z, i);
		c = vec_ele_add( c, vec_ele_mult(f, c_old) );
		_c.at(t) = c;
		c_old = _c.at(t);

		_y.at(t) = vec_ele_mult( o, tanhyp(c) );
		y_tMinus1 = _y.at(t);
	}

}

void Blstm::feed_backward(vector<vector<double>> X)
{
	/// dump the content of the cell output matrix and initialze the
	/// new cell output matrix with zeros
	_b_y.clear();
	_b_y.resize(_T, vector<double> (_hLSize, 0));
	_b_f.clear();
	_b_f.resize(_T, vector<double> (_hLSize, 0));
	_b_i.clear();
	_b_i.resize(_T, vector<double> (_hLSize, 0));
	_b_o.clear();
	_b_o.resize(_T, vector<double> (_hLSize, 0));
	_b_z.clear();
	_b_z.resize(_T, vector<double> (_hLSize, 0));
	
	_b_f_head.clear();
	_b_f_head.resize(_T, vector<double> (_hLSize, 0));
	_b_i_head.clear();
	_b_i_head.resize(_T, vector<double> (_hLSize, 0));
	_b_o_head.clear();
	_b_o_head.resize(_T, vector<double> (_hLSize, 0));
	_b_z_head.clear();
	_b_z_head.resize(_T, vector<double> (_hLSize, 0));
	
	/// dump the content of the cell matrix and initialze the new cell matrix with zeros
	_b_c.clear();
	_b_c.resize(_T, vector<double> (_hLSize, 0));
	
	vector<double> y_tPlus1(_hLSize, 0);
	vector<double> c_old(_hLSize, 0);

	/// loop from timestep t = 0 to the end of input matrix X, X(0) to X(_T)
	for (int t = _T-1; t >=0; t--)
	{
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

		f_head = vec_matrix_mult(x_t, _b_Wf);
		f_head = vec_ele_add( f_head, vec_matrix_mult(y_tPlus1, _b_Rf) );
		f_head = vec_ele_add( f_head, vec_ele_mult(_b_pf, c_old) );
		f_head = vec_ele_add( f_head, _b_bf);
		_b_f_head.at(t) = f_head;
		f = sigmoid(f_head);
		_b_f.at(t) = f;

		i_head = vec_matrix_mult(x_t, _b_Wi);
		i_head = vec_ele_add( i_head, vec_matrix_mult(y_tPlus1, _b_Ri) );
		i_head = vec_ele_add( i_head, vec_ele_mult(_b_pi, c_old) );
		i_head = vec_ele_add( i_head, _b_bi);
		_b_i_head.at(t) = i_head;
		i = sigmoid(i_head);
		_b_i.at(t) = i;

		o_head = vec_matrix_mult(x_t, _b_Wo);
		o_head = vec_ele_add( o_head, vec_matrix_mult(y_tPlus1, _b_Ro) );
		o_head = vec_ele_add( o_head, vec_ele_mult(_b_po, c_old) );
		o_head = vec_ele_add( o_head, _b_bo);
		_b_o_head.at(t) = o_head;
		o = sigmoid(o_head);
		_b_o.at(t) = o;
		
		z_head = vec_matrix_mult(x_t, _b_Wz);
		z_head = vec_ele_add( z_head, vec_matrix_mult(y_tPlus1, _b_Rz) );
		z_head = vec_ele_add( z_head, _b_bz);
		_b_z_head.at(t) = z_head;
		z = tanhyp(z_head);
		_b_z.at(t) = z;

		c = vec_ele_mult(z, i);
		c = vec_ele_add( c, vec_ele_mult(f, c_old) );
		_b_c.at(t) = c;
		c_old = _b_c.at(t);

		_b_y.at(t) = vec_ele_mult( o, tanhyp(c) );
		y_tPlus1 = _b_y.at(t);
	}

}

void Blstm::calculate_predictions()
{
	_prediction.clear();
	_prediction.resize(_T, vector<double> (_oLSize, 0));

	for (int t = 0; t < _T; t++)
	{
		vector<double> inputF(_oLSize, 0.0);
		inputF = vec_matrix_mult(_y.at(t), _Wy);
		vector<double> inputB(_oLSize, 0.0);
		inputB = vec_matrix_mult(_b_y.at(t), _b_Wy);
		vector<double> input(_oLSize, 0.0);
		input = vec_ele_add(inputF, inputB);

		_prediction.at(t) = softmax( input );
	}
}

void Blstm::calculate_single_predictions()
{
	_predictionSingle.clear();
	_predictionSingle.resize(_oLSize, 0.0);

	vector<double> inputF(_oLSize, 0.0);
	inputF = vec_matrix_mult(_y.at(_T-1), _Wy);
	vector<double> inputB(_oLSize, 0.0);
	inputB = vec_matrix_mult(_b_y.at(0), _b_Wy);
	vector<double> input(_oLSize, 0.0);
	input = vec_ele_add(inputF, inputB);

	_predictionSingle = softmax( input );
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
			//cout << _prediction.at(t).at(iOut) << endl;
			/*if (_prediction.at(t).at(iOut) < 0.001)
				L += -5 * Y.at(t).at(iOut);
			else
				L += -1 * Y.at(t).at(iOut) * log(_prediction.at(t).at(iOut));
			*/
			L += -1 * Y.at(t).at(iOut) * log(_prediction.at(t).at(iOut));
			//float div = (Y.at(t).at(iOut) - _prediction.at(t).at(iOut));
			//div = div * div;
			//L += div / 2;
		}
	}

	/// divide the loss by the length of the output train
	L = L / _T;
	//cout << "\tL: " << L;
	return L;
}

double Blstm::calculate_single_loss(vector<double> target)
{
	/// initialize the loss value with zero
	double L = 0;

	/// loop every output in Y
	for (int iOut = 0; iOut < target.size(); iOut++)
	{
		/// sum up the loss value by calculating the product Y(t)_i * log( _o(t)_i )
		/// with i = 0 .. output layer size
		//cout << _prediction.at(t).at(iOut) << endl;
		/*if (_predictionSingle.at(iOut) < 0.001)
			L += -5 * target.at(iOut);
		else
			L += -1 * target.at(iOut) * log(_predictionSingle.at(iOut));
		*/
		L += -1 * target.at(iOut) * log(_predictionSingle.at(iOut));
		//float div = (Y.at(t).at(iOut) - _prediction.at(t).at(iOut));
		//div = div * div;
		//L += div / 2;
	}

	/// divide the loss by the length of the output train
	L = L / _predictionSingle.size();
	//cout << "\tL: " << L;
	return L;
}

void Blstm::bptt(vector<vector<double>> X, vector<vector<double>> Y, vector<double> target)
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
	
	vector<double> dpi(_hLSize, 0);
	vector<double> dpf(_hLSize, 0);
	vector<double> dpo(_hLSize, 0);
	
	vector<double> dbi(_hLSize, 0);
	vector<double> dbf(_hLSize, 0);
	vector<double> dbo(_hLSize, 0);
	vector<double> dbz(_hLSize, 0);

	vector<double> dy_next(_hLSize, 0);
	vector<double> dcf_next(_hLSize, 0);
	vector<double> dpf_next(_hLSize, 0);
	vector<double> dpi_next(_hLSize, 0);
	
	vector<double> df_head(_hLSize, 0);
	vector<double> di_head(_hLSize, 0);
	vector<double> do_head(_hLSize, 0);
	vector<double> dz_head(_hLSize, 0);

	vector<double> x_t(_iLSize, 0);
	vector<double> y_t(_iLSize, 0);
		
	vector<double> deltaT(_oLSize, 0);
	deltaT = vec_ele_sub(target, _predictionSingle);
	
	/// dy(t) = delta(t) + dy(t+1)
	dy_next = vec_matrix_mult(deltaT, matrix_T(_Wy));
	dWy = matrix_add(dWy, outer(_y.at(_T-1), deltaT) );

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

		/// dpi and dpf
		dpi = vec_ele_add( dpi, vec_ele_mult(_c.at(t), di_head) );
		dpf = vec_ele_add( dpf, vec_ele_mult(_c.at(t), df_head) );

		/// Error dE/dy(t)
		/// dE/dy(t) = prediction(t) - y(t)
		//vector<double> deltaT(_oLSize, 0);
		//deltaT = vec_ele_sub(Y.at(t), _prediction.at(t));
		
		/// dy(t) = delta(t) + dy(t+1)
		//vector<double> dy(_hLSize, 0);
		//dy = vec_matrix_mult(deltaT, matrix_T(_Wy));
		//dy = vec_ele_add(dy, dy_next);

		/// do_head(t) = dy(t) * tanh( c(t) ) * dsig( o_head(t) )
		//do_head = vec_ele_mult(tanhyp(_c.at(t)), dy);
		do_head = vec_ele_mult(tanhyp(_c.at(t)), dy_next);
		do_head = vec_ele_mult(dsigmoid(_o_head.at(t)), do_head);

		/// dc(t) = dy(t) * o(t) * dtanh( c(t) ) + dc(t+1) * f(t+1)
		//vector<double> dc = vec_ele_mult( _o.at(t), dy );
		vector<double> dc = vec_ele_mult( _o.at(t), dy_next );
		dc = vec_ele_mult( dc, dtanhyp(_c.at(t)) );
		dc = vec_ele_add( dc, vec_ele_mult(_po, do_head) );
		dc = vec_ele_add( dc, dpi_next );
		dc = vec_ele_add( dc, dpf_next );
		dc = vec_ele_add( dc, dcf_next );
		
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
		
		/// dpi_next = pi * di(t+1)
		dpi_next = vec_ele_mult( _pi, di_head );
		/// dpf_next = pf * df(t+1)
		dpf_next = vec_ele_mult( _pf, df_head );
		/// dc(t+1) * f(t+1)
		dcf_next = vec_ele_mult( _f.at(t), dc );

		/// calc gradients	
		/// dWy, gradient of _Wy
		//dWy = matrix_add(dWy, outer(y_t, deltaT) );
		
		/// dWf, dWi, dWo and dWz
		dWf = matrix_add(dWf, outer(x_t, df_head));
		dWi = matrix_add(dWi, outer(x_t, di_head));
		dWo = matrix_add(dWo, outer(x_t, do_head));
		dWz = matrix_add(dWz, outer(x_t, dz_head));

		/// dbi, dbf, dbo, dbz and dpo
		dpo = vec_ele_add( dpo, vec_ele_mult(_c.at(t), do_head) );
		dbi = vec_ele_add( dbi, di_head );
		dbf = vec_ele_add( dbf, df_head );
		dbo = vec_ele_add( dbo, do_head );
		dbz = vec_ele_add( dbz, dz_head );

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
	
	//double L = calculate_loss(Y);
	double L = calculate_single_loss(target);

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

	_pi = vec_ele_add_with_const(_pi, dpi, L);
	_pf = vec_ele_add_with_const(_pf, dpf, L);
	_po = vec_ele_add_with_const(_po, dpo, L);

	_bi = vec_ele_add_with_const(_bi, dbi, L);
	_bf = vec_ele_add_with_const(_bf, dbf, L);
	_bo = vec_ele_add_with_const(_bo, dbo, L);
	_bz = vec_ele_add_with_const(_bz, dbz, L);
}

void Blstm::fptt(vector<vector<double>> X, vector<vector<double>> Y, vector<double> target)
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
	
	vector<double> dpi(_hLSize, 0);
	vector<double> dpf(_hLSize, 0);
	vector<double> dpo(_hLSize, 0);
	
	vector<double> dbi(_hLSize, 0);
	vector<double> dbf(_hLSize, 0);
	vector<double> dbo(_hLSize, 0);
	vector<double> dbz(_hLSize, 0);

	vector<double> dy_next(_hLSize, 0);
	vector<double> dcf_next(_hLSize, 0);
	vector<double> dpf_next(_hLSize, 0);
	vector<double> dpi_next(_hLSize, 0);
	
	vector<double> df_head(_hLSize, 0);
	vector<double> di_head(_hLSize, 0);
	vector<double> do_head(_hLSize, 0);
	vector<double> dz_head(_hLSize, 0);

	vector<double> x_t(_iLSize, 0);
	vector<double> y_t(_iLSize, 0);
		
	/// Error dE/dy(t)
	/// dE/dy(t) = prediction(t) - y(t)
	vector<double> deltaT(_oLSize, 0);
	deltaT = vec_ele_sub(target, _predictionSingle);
	
	/// dy(t) = delta(t) + dy(t+1)
	dy_next = vec_matrix_mult(deltaT, matrix_T(_b_Wy));
	dWy = matrix_add(dWy, outer(_b_y.at(0), deltaT) );

	/// loop every output forward from 0 to _t -1
	for (int t = 0; t < _T; t++)
	{
		y_t = _b_y.at(t);
		x_t = X.at(t);

		/// dRf, dRi, dRo and dRz
		dRf = matrix_add(dRf, outer(y_t, df_head));
		dRi = matrix_add(dRi, outer(y_t, di_head));
		dRo = matrix_add(dRo, outer(y_t, do_head));
		dRz = matrix_add(dRz, outer(y_t, dz_head));

		/// dpi and dpf
		dpi = vec_ele_add( dpi, vec_ele_mult(_b_c.at(t), di_head) );
		dpf = vec_ele_add( dpf, vec_ele_mult(_b_c.at(t), df_head) );

		/// Error dE/dy(t)
		/// dE/dy(t) = prediction(t) - y(t)
		//vector<double> deltaT(_oLSize, 0);
		//deltaT = vec_ele_sub(Y.at(t), _prediction.at(t));
		
		/// dy(t) = delta(t) + dy(t+1)
		//vector<double> dy(_hLSize, 0);
		//dy = vec_matrix_mult(deltaT, matrix_T(_b_Wy));
		//dy = vec_ele_add(dy, dy_next);

		/// do_head(t) = dy(t) * tanh( c(t) ) * dsig( o_head(t) )
		//do_head = vec_ele_mult(tanhyp(_b_c.at(t)), dy);
		do_head = vec_ele_mult(tanhyp(_b_c.at(t)), dy_next);
		do_head = vec_ele_mult(dsigmoid(_b_o_head.at(t)), do_head);

		/// dc(t) = dy(t) * o(t) * dtanh( c(t) ) + dc(t+1) * f(t+1)
		//vector<double> dc = vec_ele_mult( _b_o.at(t), dy );
		vector<double> dc = vec_ele_mult( _b_o.at(t), dy_next );
		dc = vec_ele_mult( dc, dtanhyp(_b_c.at(t)) );
		dc = vec_ele_add( dc, vec_ele_mult(_b_po, do_head) );
		dc = vec_ele_add( dc, dpi_next );
		dc = vec_ele_add( dc, dpf_next );
		dc = vec_ele_add( dc, dcf_next );
		
		/// df_head(t) = dc(t) * c(t+1) * dsig( f_head(t) )
		if (t < (_T-1))
			df_head = vec_ele_mult(_b_c.at(t+1), dc);
		df_head = vec_ele_mult(dsigmoid(_b_f_head.at(t)), df_head);	
		
		/// di_head(t) = dc(t) * z(t) * dsig( i_head(t) )
		di_head = vec_ele_mult(_b_z.at(t), dc);
		di_head = vec_ele_mult(dsigmoid(_b_i_head.at(t)), di_head);

		/// dz_head(t) = dc(t) * i(t) * dtanh( _z_head(t) )
		dz_head = vec_ele_mult(_b_i.at(t), dc);
		dz_head = vec_ele_mult(dtanhyp(_b_z_head.at(t)), dz_head);
		
		/// dpi_next = pi * di(t+1)
		dpi_next = vec_ele_mult( _b_pi, di_head );
		/// dpf_next = pf * df(t+1)
		dpf_next = vec_ele_mult( _b_pf, df_head );
		/// dc(t+1) * f(t+1)
		dcf_next = vec_ele_mult( _b_f.at(t), dc );

		/// calc gradients	
		/// dWy, gradient of _Wy
		//dWy = matrix_add(dWy, outer(y_t, deltaT) );
		
		/// dWf, dWi, dWo and dWz
		dWf = matrix_add(dWf, outer(x_t, df_head));
		dWi = matrix_add(dWi, outer(x_t, di_head));
		dWo = matrix_add(dWo, outer(x_t, do_head));
		dWz = matrix_add(dWz, outer(x_t, dz_head));

		/// dbi, dbf, dbo, dbz and dpo
		dpo = vec_ele_add( dpo, vec_ele_mult(_b_c.at(t), do_head) );
		dbi = vec_ele_add( dbi, di_head );
		dbf = vec_ele_add( dbf, df_head );
		dbo = vec_ele_add( dbo, do_head );
		dbz = vec_ele_add( dbz, dz_head );

		/// dy_next
		vector<double> dy_Rf = vec_matrix_mult(df_head, matrix_T(_b_Rf));
		vector<double> dy_Ri = vec_matrix_mult(di_head, matrix_T(_b_Ri));
		vector<double> dy_Ro = vec_matrix_mult(do_head, matrix_T(_b_Ro));
		vector<double> dy_Rz = vec_matrix_mult(dz_head, matrix_T(_b_Rz));
		
		vector<double> dy_R = vec_ele_add(dy_Ro, dy_Rz);
		dy_R = vec_ele_add(dy_R, dy_Ri);
		dy_R = vec_ele_add(dy_R, dy_Rf);
		dy_next = dy_R;
	}
	
	//double L = calculate_loss(Y);
	double L = calculate_single_loss(target);

	L = L * _learningRate;
	
	_b_Wy = matrix_add_with_const(_b_Wy, dWy, L);
	_b_Wf = matrix_add_with_const(_b_Wf, dWf, L);
	_b_Wi = matrix_add_with_const(_b_Wi, dWi, L);
	_b_Wo = matrix_add_with_const(_b_Wo, dWo, L);
	_b_Wz = matrix_add_with_const(_b_Wz, dWz, L);

	_b_Rf = matrix_add_with_const(_b_Rf, dRf, L);
	_b_Ri = matrix_add_with_const(_b_Ri, dRi, L);
	_b_Ro = matrix_add_with_const(_b_Ro, dRo, L);
	_b_Rz = matrix_add_with_const(_b_Rz, dRz, L);

	_b_pi = vec_ele_add_with_const(_b_pi, dpi, L);
	_b_pf = vec_ele_add_with_const(_b_pf, dpf, L);
	_b_po = vec_ele_add_with_const(_b_po, dpo, L);

	_b_bi = vec_ele_add_with_const(_b_bi, dbi, L);
	_b_bf = vec_ele_add_with_const(_b_bf, dbf, L);
	_b_bo = vec_ele_add_with_const(_b_bo, dbo, L);
	_b_bz = vec_ele_add_with_const(_b_bz, dbz, L);
}

void Blstm::random_weights()
{
	///	give the random generator a seed
	srand(time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);

	/// normal distribution, mean = 0.0, deviation = 0.1
	std::normal_distribution<double> distribution (0.0,0.3);
	
	/// loop every element in matrix _Wf and assign a random value
	for (int m = 0; m < _Wf.size() ; m++)
	{
		for (int n = 0; n < _Wf.at(m).size() ; n++)
		{
			_Wf.at(m).at(n) = distribution(generator);
			_b_Wf.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Wi and assign a random value
	for (int m = 0; m < _Wi.size() ; m++)
	{
		for (int n = 0; n < _Wi.at(m).size() ; n++)
		{
			_Wi.at(m).at(n) = distribution(generator);
			_b_Wi.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Wz and assign a random value
	for (int m = 0; m < _Wz.size() ; m++)
	{
		for (int n = 0; n < _Wz.at(m).size() ; n++)
		{
			_Wz.at(m).at(n) = distribution(generator);
			_b_Wz.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Wo and assign a random value
	for (int m = 0; m < _Wo.size() ; m++)
	{
		for (int n = 0; n < _Wo.at(m).size() ; n++)
		{
			_Wo.at(m).at(n) = distribution(generator);
			_b_Wo.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Wy and assign a random value
	for (int m = 0; m < _Wy.size() ; m++)
	{
		for (int n = 0; n < _Wy.at(m).size() ; n++)
		{
			_Wy.at(m).at(n) = distribution(generator);
			_b_Wy.at(m).at(n) = distribution(generator);
		}
	}
	
	/// Recurrent weights
	/// loop every element in matrix _Rf and assign a random value
	for (int m = 0; m < _Rf.size() ; m++)
	{
		for (int n = 0; n < _Rf.at(m).size() ; n++)
		{
			_Rf.at(m).at(n) = distribution(generator);
			_b_Rf.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Ri and assign a random value
	for (int m = 0; m < _Ri.size() ; m++)
	{
		for (int n = 0; n < _Ri.at(m).size() ; n++)
		{
			_Ri.at(m).at(n) = distribution(generator);
			_b_Ri.at(m).at(n) = distribution(generator);
		}
	}

	/// loop every element in matrix _Rz and assign a random value
	for (int m = 0; m < _Rz.size() ; m++)
	{
		for (int n = 0; n < _Rz.at(m).size() ; n++)
		{
			_Rz.at(m).at(n) = distribution(generator);
			_b_Rz.at(m).at(n) = distribution(generator);
		}
	}
	
	/// loop every element in matrix _Ro and assign a random value
	for (int m = 0; m < _Ro.size() ; m++)
	{
		for (int n = 0; n < _Ro.at(m).size() ; n++)
		{
			_Ro.at(m).at(n) = distribution(generator);
			_b_Ro.at(m).at(n) = distribution(generator);
		}
	}

	/// Bias weights
	/// _bi
	for (int i = 0; i < _bi.size(); i++)
	{
		_bi.at(i) = distribution(generator);
		_b_bi.at(i) = distribution(generator);
	}

	/// _bo
	for (int i = 0; i < _bo.size(); i++)
	{
		_bo.at(i) = distribution(generator);
		_b_bo.at(i) = distribution(generator);
	}

	/// _bf
	for (int i = 0; i < _bf.size(); i++)
	{
		_bf.at(i) = distribution(generator);
		_b_bf.at(i) = distribution(generator);
	}

	/// _bz
	for (int i = 0; i < _bz.size(); i++)
	{
		_bz.at(i) = distribution(generator);
		_b_bz.at(i) = distribution(generator);
	}

	/// Peephole weights
	/// _pi
	for (int i = 0; i < _pi.size(); i++)
	{
		_pi.at(i) = distribution(generator);
		_b_pi.at(i) = distribution(generator);
	}

	/// _pf
	for (int i = 0; i < _pf.size(); i++)
	{
		_pf.at(i) = distribution(generator);
		_b_pf.at(i) = distribution(generator);
	}

	/// _po
	for (int i = 0; i < _po.size(); i++)
	{
		_po.at(i) = distribution(generator);
		_b_po.at(i) = distribution(generator);
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
	//print_matrix("_Rz", _Rz);
}

void Blstm::render_weights(int index)
{	
	/// render recurrent weights
	unsigned int heightR, widthR;
	/// _Rz
	vector<double> rz;
	vector<vector<double>> ampRz(_Rz.size(), vector<double>(_Rz.at(0).size(), 0));
	ampRz = matrix_add_with_const(ampRz, _Rz, 3);
	matrix_to_vector(ampRz, heightR, widthR, rz);
	render::vector_to_PNG("_Rz", std::to_string(index), "exp", heightR, widthR, rz);

	/// _Ri
	vector<double> ri;
	vector<vector<double>> ampRi(_Ri.size(), vector<double>(_Ri.at(0).size(), 0));
	ampRi = matrix_add_with_const(ampRi, _Ri, 3);
	matrix_to_vector(ampRi, heightR, widthR, ri);
	render::vector_to_PNG("_Ri", std::to_string(index), "exp", heightR, widthR, ri);

	/// _Rf
	vector<double> rf;
	vector<vector<double>> ampRf(_Rf.size(), vector<double>(_Rf.at(0).size(), 0));
	ampRf = matrix_add_with_const(ampRf, _Rf, 3);
	matrix_to_vector(ampRf, heightR, widthR, rf);
	render::vector_to_PNG("_Rf", std::to_string(index), "exp", heightR, widthR, rf);

	/// _Ro
	vector<double> ro;
	vector<vector<double>> ampRo(_Ro.size(), vector<double>(_Ro.at(0).size(), 0));
	ampRo = matrix_add_with_const(ampRo, _Ro, 3);
	matrix_to_vector(ampRo, heightR, widthR, ro);
	render::vector_to_PNG("_Ro", std::to_string(index), "exp", heightR, widthR, ro);

}

bool Blstm::check_weight_sum()
{
	bool status = true;

	float sumRz = matrix_sum(_Rz);
	float sumRo = matrix_sum(_Ro);
	float sumRi = matrix_sum(_Ri);
	float sumRf = matrix_sum(_Rf);
	float sumWz = matrix_sum(_Wz);
	float sumWi = matrix_sum(_Wi);
	float sumWo = matrix_sum(_Wo);
	float sumWf = matrix_sum(_Wf);
	float sumWy = matrix_sum(_Wy);

	cout << "\tSUM Rz: " << sumRz << "\tRo: " << sumRo
		<< "\tRi: " << sumRi << "\tRf: " << sumRf
		<< "\tWo: " << sumWo << "\tWz: " << sumWz
		<< "\tWi: " << sumWi << "\tWf: " << sumWf
		<< "\tWy: " << sumWy << endl;

	float limit = 500.0;
	
	if ( (sumRz > limit) || (sumRo > limit) || (sumRi > limit) || (sumRf > limit) 
			|| (sumWz > limit) || (sumWi > limit) || (sumWo > limit) || (sumWf > limit)
			|| (sumWy > limit) )
	{
		status = false;
	}

	return status;
}

vector<unsigned> Blstm::get_topo()
{
	vector<unsigned> topology;
	topology.push_back((unsigned)_iLSize);
	topology.push_back((unsigned)_hLSize);
	topology.push_back((unsigned)_oLSize);

	return topology;
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

	/// _b_Wy
	outputFile << "bWy" << endl;
	/// get matrix dimensions
	height = _b_Wy.size();
	width = _b_Wy.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Wy.at(i).at(j) << ' ';
			else
				outputFile << _b_Wy.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Wi
	outputFile << "Wi" << endl;
	/// get matrix dimensions
	height = _Wi.size();
	width = _Wi.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Wi.at(i).at(j) << ' ';
			else
				outputFile << _Wi.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Wi
	outputFile << "bWi" << endl;
	/// get matrix dimensions
	height = _b_Wi.size();
	width = _b_Wi.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Wi.at(i).at(j) << ' ';
			else
				outputFile << _b_Wi.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Wf
	outputFile << "Wf" << endl;
	/// get matrix dimensions
	height = _Wf.size();
	width = _Wf.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Wf.at(i).at(j) << ' ';
			else
				outputFile << _Wf.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Wf
	outputFile << "bWf" << endl;
	/// get matrix dimensions
	height = _b_Wf.size();
	width = _b_Wf.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Wf.at(i).at(j) << ' ';
			else
				outputFile << _b_Wf.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Wo
	outputFile << "Wo" << endl;
	/// get matrix dimensions
	height = _Wo.size();
	width = _Wo.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Wo.at(i).at(j) << ' ';
			else
				outputFile << _Wo.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Wo
	outputFile << "bWo" << endl;
	/// get matrix dimensions
	height = _b_Wo.size();
	width = _b_Wo.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Wo.at(i).at(j) << ' ';
			else
				outputFile << _b_Wo.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Wz
	outputFile << "Wz" << endl;
	/// get matrix dimensions
	height = _Wz.size();
	width = _Wz.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Wz.at(i).at(j) << ' ';
			else
				outputFile << _Wz.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Wz
	outputFile << "bWz" << endl;
	/// get matrix dimensions
	height = _b_Wz.size();
	width = _b_Wz.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Wz.at(i).at(j) << ' ';
			else
				outputFile << _b_Wz.at(i).at(j);
		}
		outputFile << endl;
	}

	/// Recursive weights
	/// _Ri
	outputFile << "Ri" << endl;
	/// get matrix dimensions
	height = _Ri.size();
	width = _Ri.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Ri.at(i).at(j) << ' ';
			else
				outputFile << _Ri.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Ri
	outputFile << "bRi" << endl;
	/// get matrix dimensions
	height = _b_Ri.size();
	width = _b_Ri.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Ri.at(i).at(j) << ' ';
			else
				outputFile << _b_Ri.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Rf
	outputFile << "Rf" << endl;
	/// get matrix dimensions
	height = _Rf.size();
	width = _Rf.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Rf.at(i).at(j) << ' ';
			else
				outputFile << _Rf.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Rf
	outputFile << "bRf" << endl;
	/// get matrix dimensions
	height = _b_Rf.size();
	width = _b_Rf.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Rf.at(i).at(j) << ' ';
			else
				outputFile << _b_Rf.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Ro
	outputFile << "Ro" << endl;
	/// get matrix dimensions
	height = _Ro.size();
	width = _Ro.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Ro.at(i).at(j) << ' ';
			else
				outputFile << _Ro.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _b_Ro
	outputFile << "bRo" << endl;
	/// get matrix dimensions
	height = _b_Ro.size();
	width = _b_Ro.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Ro.at(i).at(j) << ' ';
			else
				outputFile << _b_Ro.at(i).at(j);
		}
		outputFile << endl;
	}

	/// _Rz
	outputFile << "Rz" << endl;
	/// get matrix dimensions
	height = _Rz.size();
	width = _Rz.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _Rz.at(i).at(j) << ' ';
			else
				outputFile << _Rz.at(i).at(j);
		}
		outputFile << endl;
	}	

	/// _b_Rz
	outputFile << "bRz" << endl;
	/// get matrix dimensions
	height = _b_Rz.size();
	width = _b_Rz.at(0).size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j < (width - 1))
				outputFile << _b_Rz.at(i).at(j) << ' ';
			else
				outputFile << _b_Rz.at(i).at(j);
		}
		outputFile << endl;
	}	

	/// Bias
	/// _bi
	outputFile << "bi" << endl;
	///	write matrix values to file
	for (int i = 0; i < _bi.size(); i++)
	{
		if (i < (_bi.size() - 1))
			outputFile << _bi.at(i) << ' ';
		else
			outputFile << _bi.at(i);
	}
	outputFile << endl;

	/// _b_bi
	outputFile << "bbi" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_bi.size(); i++)
	{
		if (i < (_b_bi.size() - 1))
			outputFile << _b_bi.at(i) << ' ';
		else
			outputFile << _b_bi.at(i);
	}
	outputFile << endl;

	/// _bf
	outputFile << "bf" << endl;
	///	write matrix values to file
	for (int i = 0; i < _bf.size(); i++)
	{
		if (i < (_bf.size() - 1))
			outputFile << _bf.at(i) << ' ';
		else
			outputFile << _bf.at(i);
	}
	outputFile << endl;

	/// _b_bf
	outputFile << "bbf" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_bf.size(); i++)
	{
		if (i < (_b_bf.size() - 1))
			outputFile << _b_bf.at(i) << ' ';
		else
			outputFile << _b_bf.at(i);
	}
	outputFile << endl;

	/// _bo
	outputFile << "bo" << endl;
	///	write matrix values to file
	for (int i = 0; i < _bo.size(); i++)
	{
		if (i < (_bo.size() - 1))
			outputFile << _bo.at(i) << ' ';
		else
			outputFile << _bo.at(i);
	}
	outputFile << endl;

	/// _b_bo
	outputFile << "bbo" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_bo.size(); i++)
	{
		if (i < (_b_bo.size() - 1))
			outputFile << _b_bo.at(i) << ' ';
		else
			outputFile << _b_bo.at(i);
	}
	outputFile << endl;

	/// _bz
	outputFile << "bz" << endl;
	///	write matrix values to file
	for (int i = 0; i < _bz.size(); i++)
	{
		if (i < (_bz.size() - 1))
			outputFile << _bz.at(i) << ' ';
		else
			outputFile << _bz.at(i);
	}
	outputFile << endl;

	/// _b_bz
	outputFile << "bbz" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_bz.size(); i++)
	{
		if (i < (_b_bz.size() - 1))
			outputFile << _b_bz.at(i) << ' ';
		else
			outputFile << _b_bz.at(i);
	}
	outputFile << endl;

	/// _pi
	outputFile << "pi" << endl;
	///	write matrix values to file
	for (int i = 0; i < _pi.size(); i++)
	{
		if (i < (_pi.size() - 1))
			outputFile << _pi.at(i) << ' ';
		else
			outputFile << _pi.at(i);
	}
	outputFile << endl;

	/// _b_pi
	outputFile << "bpi" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_pi.size(); i++)
	{
		if (i < (_b_pi.size() - 1))
			outputFile << _b_pi.at(i) << ' ';
		else
			outputFile << _b_pi.at(i);
	}
	outputFile << endl;

	/// _pf
	outputFile << "pf" << endl;
	///	write matrix values to file
	for (int i = 0; i < _pf.size(); i++)
	{
		if (i < (_pf.size() - 1))
			outputFile << _pf.at(i) << ' ';
		else
			outputFile << _pf.at(i);
	}
	outputFile << endl;

	/// _b_pf
	outputFile << "bpf" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_pf.size(); i++)
	{
		if (i < (_b_pf.size() - 1))
			outputFile << _b_pf.at(i) << ' ';
		else
			outputFile << _b_pf.at(i);
	}
	outputFile << endl;

	/// _po
	outputFile << "po" << endl;
	///	write matrix values to file
	for (int i = 0; i < _po.size(); i++)
	{
		if (i < (_po.size() - 1))
			outputFile << _po.at(i) << ' ';
		else
			outputFile << _po.at(i);
	}
	outputFile << endl;

	/// _b_po
	outputFile << "bpo" << endl;
	///	write matrix values to file
	for (int i = 0; i < _b_po.size(); i++)
	{
		if (i < (_b_po.size() - 1))
			outputFile << _b_po.at(i) << ' ';
		else
			outputFile << _b_po.at(i);
	}
	outputFile << endl;

	///	close file
	outputFile.close();
}
