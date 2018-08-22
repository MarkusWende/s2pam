#include <iostream>
#include <iomanip> // std::setprecision
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <algorithm>
#include <math.h>
#include <experimental/filesystem>
#include <thread>
#include <chrono>
#include <future>
#include <mutex>

#include "render.h"
#include "helper.h"
#include "credit_libav.h"
#include "ThreadPool.h"
#include "wave_read.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using hires_clock = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;
using namespace std::this_thread;     // sleep_for, sleep_until
using namespace std::chrono_literals; // ns, us, ms, s, h, etc.

namespace fs = std::experimental::filesystem;
std::mutex mtx;

int processTrainingAudioFile(string audioFilename) {

	cout << "--Computing: " << audioFilename << endl;

	/////// PARAMS //////////////
	int sampleRate = 16000;
	int frameSize = 1024;
	int hopSize = 64;

	AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
	
	Wave wave(audioFilename.c_str());

	int qSteps = pow(2,wave.bitsPerSample());
	vector<Real> audioBuffer(wave.numberOfSamples());

	for (int i=0; i<wave.numberOfSamples(); i++) {
		float sample;
		sample = static_cast<float>(wave.samples[i]) / qSteps;
		audioBuffer[i] = sample;
	}

	Algorithm* fc    = factory.create("FrameCutter",
																		"frameSize", frameSize,						// default: 1024
																		"hopSize", hopSize,								// default: 512
																		//"lastFrameToEndOfFile", false,	// default: false
																		//"startFromZero", false,					// default: false
																		"validFrameThresholdRatio", 0			// default: 0
																		);

	Algorithm* w     = factory.create("Windowing",
																		//"normalized", true,							// default: true
																		"size", 64,												// default: 1024
																		"zeroPadding", frameSize+hopSize,	// default: 0
																		"type", "hann",										// default: "hann"
																		"zeroPhase", true									// default: true
																		);

	Algorithm* spec  = factory.create("Spectrum",
																		"size", 2048											// default: 2048
																		);

	Algorithm* mfcc  = factory.create("MFCC",
																		//"dctType", 2,										// default: 2
																		//"highFrequencyBound", 11000,		// default: 11000
																		//"inputSize", 1025,							// default: 1025
																		//"liftering", 0,									// default: 0
																		//"logType", "dbamp",							// default: "dbamp"
																		//"lowFrequencyBound", 0,					// default: 0
																		//"normalize", "unit_sum",				// default: "unit_max"
																		"numberBands", 40,								// default: 40
																		"numberCoefficients", 13,					// default: 13
																		//"sampleRate", 44100,						// default: 44100
																		//"type", "magnitude",						// default: "magnitude"
																		//"warpingFormula", "slaneyMel",	// default: "slaneyMel"
																		"weighting", "warping"						// default: "warping"
																		);

	/////////// CONNECTING THE ALGORITHMS ////////////////
	E_INFO("-------- connecting algos ---------");

	fc->input("signal").set(audioBuffer);

	// FrameCutter -> Windowing -> Spectrum
	std::vector<Real> frame, windowedFrame;

	fc->output("frame").set(frame);
	w->input("frame").set(frame);

	w->output("frame").set(windowedFrame);
	spec->input("frame").set(windowedFrame);

	// Spectrum -> MFCC
	std::vector<Real> spectrum, mfccCoeffs, mfccBands;

	spec->output("spectrum").set(spectrum);
	mfcc->input("spectrum").set(spectrum);

	mfcc->output("bands").set(mfccBands);
	mfcc->output("mfcc").set(mfccCoeffs);

	/////////// STARTING THE ALGORITHMS //////////////////
	E_INFO("-------- start processing " << audioFilename << " --------");

	// declare matrix
	vector<vector<float>> mSpectrum;
	vector<vector<float>> mMfccCoeffs;
	vector<vector<float>> mMfccBands;

	int counter = 0;
	while (true) {

		// compute a frame
		fc->compute();

		// if it was the last one (ie: it was empty), then we're done.
		if (!frame.size()) {
			break;
		}

		// if the frame is silent, just drop it and go on processing
		if (isSilent(frame)) continue;

		w->compute();
		spec->compute();
		mfcc->compute();
		
		// make new row (arbitrary example)
		vector<Real> spectrogramRow(1,spectrum.size());
		mSpectrum.push_back(spectrogramRow);
		
		for (std::vector<Real>::iterator it = spectrum.begin(); it != spectrum.end(); ++it) {
			// add element to row
			mSpectrum[counter].push_back(*it);
		}

		vector<Real> mfccCoeffsRow(1,mfccCoeffs.size());
		mMfccCoeffs.push_back(mfccCoeffsRow);
		for (std::vector<Real>::iterator it = mfccCoeffs.begin(); it != mfccCoeffs.end(); ++it) {
			// add element to row
			mMfccCoeffs[counter].push_back(*it);
		}

		vector<Real> mfccBandsRow(1,mfccBands.size());
		mMfccBands.push_back(mfccBandsRow);
		for (std::vector<Real>::iterator it = mfccBands.begin(); it != mfccBands.end(); ++it) {
			// add element to row
			mMfccBands[counter].push_back(*it);
		}

		counter++;
	}

	vector<float> vSpectrumNormalized;

	vector<vector<float>> mMfccCoeffsEnlarged(mSpectrum.size(),vector<float>(mSpectrum[0].size(),0));
	vector<float> vMfccCoeffsNormalized;

	vector<vector<float>> mMfccBandsEnlarged(mSpectrum.size(),vector<float>(mSpectrum[0].size(),0));
	vector<float> vMfccBandsNormalized;

	unsigned int imageHeight;
	unsigned int imageWidth;


	// generate png's from matrixes
	helper::matrix_to_normalized_vector(mSpectrum, imageHeight, imageWidth, vSpectrumNormalized);
	render::vector_to_PNG(audioFilename, "_spec", imageHeight, imageWidth, vSpectrumNormalized);

	helper::matrix_enlarge(mMfccBands, mMfccBandsEnlarged);
	helper::matrix_to_normalized_vector(mMfccBandsEnlarged, imageHeight, imageWidth, vMfccBandsNormalized);
	render::vector_to_PNG(audioFilename, "_bands", imageHeight, imageWidth, vMfccBandsNormalized);

	helper::matrix_enlarge(mMfccCoeffs, mMfccCoeffsEnlarged);
	helper::matrix_to_normalized_vector(mMfccCoeffsEnlarged, imageHeight, imageWidth, vMfccCoeffsNormalized);
	render::vector_to_PNG(audioFilename, "_mfcc", imageHeight, imageWidth, vMfccCoeffsNormalized);

	// generate mfcc file from matrix
	render::matrix_to_MFCC_file(mMfccCoeffs, audioFilename);

	// clear memory
	delete fc;
	delete w;
	delete spec;
	delete mfcc;

	return 1;
}

int main(int argc, char* argv[]) {

  auto t1 = hires_clock::now();
	
	if (argc < 2) {
    E_ERROR("\tIncorrect number of arguments.\n" <<
			"\t\tUsage: " << argv[0] << " [OPTION]... PATH_TO_TRAINING_SET"
			<< "\n\n\t\tOptions:\n" << "\t\t\t\t-v\t\tVerbose mode.\n");
    exit(1);
  }

	// check if audio files for training do exist
	if (!fs::is_directory("./data/TIMIT/Audio") || !fs::exists("./data/TIMIT/Audio")) {
		E_ERROR("\tNo training data folder found.\n" <<
			"\t\tCreate ./data/TIMIT/Audio with training audio files inside.");
		exit(1);
	}

  string trainingDataPath = argv[1];

	// set the logging level
	if (argc > 2) {
		string argVerbose = "-v";
		if(argVerbose.compare(argv[1]) == 0) {
			infoLevelActive = true;
		} else {
			infoLevelActive = false;
		}
		trainingDataPath = argv[2];
	} else {
		infoLevelActive = false;
	}

	// create folder if it doesnt exist
	if (!fs::is_directory("./data/SPECS") || !fs::exists("./data/SPECS")) {
		E_INFO("\tCreating directory ./data/SPECS");
		fs::create_directory("./data/SPECS");
	}
	if (!fs::is_directory("./data/FEATURES") || !fs::exists("./data/FEATURES")) {
		E_INFO("\tCreating directory ./data/FEATURES");
		fs::create_directory("./data/FEATURES");
	}


	fs::directory_iterator trainingDataPathIter(trainingDataPath), e;
	std::vector<fs::path> trainingFileList(trainingDataPathIter, e);


	string fileName;
	bool finished = false;
	int totalCount = 0;
	while(!finished) {

		// register the algorithms in the factory(ies)
		essentia::init();
	
		ThreadPool pool(4);
		std::vector< std::future<int> > results;

		int tmpCount = 0;
		int remainCount = 0;

		if((trainingFileList.size()-totalCount) >=100)
			remainCount = 100;
		else
			remainCount = trainingFileList.size()-totalCount;

		for(int i = 0; i < remainCount; i++) {
		//for(int i = 0; i < 1; i++) {
		
			fileName = trainingFileList[i+totalCount];
			
			results.emplace_back(pool.enqueue(processTrainingAudioFile, fileName));
			//sleep_for(100ms);	
		}

		for(auto && result: results)
			result.get();

		totalCount = totalCount + remainCount;
		if(totalCount >= trainingFileList.size()) {
			finished = true;
		}

		printf("------------------------------ 100 RUNs -----------------------------------\n");
		essentia::shutdown();

		sleep_for(500ms);	
	
		//return 0;
	}


  std::cout << "Elapsed: " << duration_ms(hires_clock::now() - t1).count() << " ms\n";
	
	return 0;
}
