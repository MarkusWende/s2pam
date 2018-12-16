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
		double _accSum;
		double _population;

		std::vector<double> _TP;
		std::vector<double> _TN;
		std::vector<double> _FP;
		std::vector<double> _FN;
		std::vector<double> _total;
		std::vector<std::string> _labels;
		std::string _labelType;

		std::vector<double> _A;
		std::vector<double> _P;
		std::string _AString;
		std::string _PString;
		std::vector<std::vector<double>> _AP;
		std::vector<std::vector<std::string>> _APString;

	public:
		
		Statistics(
				int numClass,
				std::string type,
				double p
				);

		void process(
				std::vector<double> A,
				std::vector<double> P
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

		std::vector<double> get_fScore() { return _fScore; };
		std::vector<double> get_acc() { return _acc; };
		std::vector<std::vector<double>> get_confMat() { return _confusionMatrix; };
		std::vector<std::vector<double>> get_AP() { return _AP; };
		std::vector<std::vector<std::string>> get_APString() { return _APString; };

		void print_all();
		void concat_AP_binary();
		void concat_AP();
		std::string get_string_representation_vc(
				std::vector<double> binIn
				);
		
		std::string get_string_representation_phn(
				std::vector<double> binIn
				);
		
		std::string get_string_representation_art(
				std::vector<double> binIn
				);
};
#endif		// SATISTIC_H
