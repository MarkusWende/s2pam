/**
 * @file	blstm.h
 * @class	BLSTM
 *
 * @brief	BLSTM (Bidirectional long short term memory) Neural Network
 *
 *			This class represents the blstm cell structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef BLSTM_H
#define BLSTM_H

#include<vector>
#include<math.h>
#include<cassert>

#include "cell.h"

		
class Blstm
{
	private:
		std::vector<Layer> layers_; // m_layers[layerNum][cellNum]
		double error_;
		double recentAverageError_;
		double recentAverageSmoothingFactor_;

	public:
		Blstm(const std::vector<unsigned> &topology);
		void feed_forward(const std::vector<double> &inputVals);
		void back_prop(const std::vector<double> &targetVals);
		void get_results(std::vector<double> &resultVals) const;
		double get_recent_average_error(void) const { return recentAverageError_; };
		std::vector<Layer> get_layers() { return layers_; };

};			// end of class BLSTM
#endif		// BLSTM_H
