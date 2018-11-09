/**
 * @file	statistic.h
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

#ifndef STATISTIC_H
#define STATISTIC_H

#include <vector>

#include "helper.h"

class Statistics
{
	private:
		std::vector<std::vector<double>> _confusionMatrix;
		std::vector<double> _fScore;
		std::vector<double> _precision;
		std::vector<double> _recall;
		std::vector<double> _acc;
		std::vector<double> _classError;
		double _fScoreSum;
		double _accSum;

		std::vector<double> _TP;
		std::vector<double> _TN;
		std::vector<double> _FP;
		std::vector<double> _FN;
		std::vector<double> _total;

		std::vector<std::vector<double>> _A;
		std::vector<std::vector<double>> _P;

	public:
		
		Statistics(
				int numClass
				);

		void process(
				std::vector<std::vector<double>> A,
				std::vector<std::vector<double>> P
				);

		void true_positive();
		void true_negative();
		void false_negative();
		void false_positive();

		void precision();
		void recall();
		void accuracy();
		void fScore();
		
		void confusion_matrix();

		void print_all();
};
#endif		// SATISTIC_H
