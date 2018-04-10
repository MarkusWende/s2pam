#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <algorithm>
#include <math.h>

#include "render.h"
#include "helper.h"
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;


int main(int argc, char* argv[]) {

  if (argc != 2) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  //string outputFilename = argv[2];

// register the algorithms in the factory(ies)
essentia::init();

Pool pool;

/////// PARAMS //////////////
int sampleRate = 12000;
int frameSize = 1024;
int hopSize = 20;

AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

Algorithm* audio = factory.create("MonoLoader",
																	//"audioStream", 0,										// default: 0
																	//"downmix", "mix",										// default: "mix"
                                  "filename", audioFilename,						//
                                  "sampleRate", sampleRate							// default: 44100
																	);

Algorithm* fc    = factory.create("FrameCutter",
                                  "frameSize", frameSize,								// default: 1024
                                  "hopSize", hopSize,										// default: 512
																	//"lastFrameToEndOfFile", false,			// default: false
																	//"startFromZero", false,							// default: false
																	"validFrameThresholdRatio", 0					// default: 0
																	);

Algorithm* w     = factory.create("Windowing",
																	//"normalized", true,									// default: true
																	"size", 64,														// default: 1024
																	"zeroPadding", 512+frameSize,					// default: 0
                                  "type", "blackmanharris62",						// default: "hann"
																	"zeroPhase", true											// default: true
																	);

Algorithm* spec  = factory.create("Spectrum",
																	"size", 2048													// default: 2048
																	);
Algorithm* mfcc  = factory.create("MFCC",
																	//"dctType", 2,												// default: 2
																	//"highFrequencyBound", 11000,				// default: 11000
																	//"inputSize", 1025,									// default: 1025
																	//"liftering", 0,											// default: 0
																	//"logType", "dbamp",									// default: "dbamp"
																	//"lowFrequencyBound", 0,							// default: 0
																	//"normalize", "unit_max",						// default: "unit_max"
																	"numberBands", 90,										// default: 40
																	"numberCoefficients", 45,							// default: 13
																	//"sampleRate", 44100,								// default: 44100
																	//"type", "magnitude",								// default: "magnitude"
																	//"warpingFormula", "slaneyMel",			// default: "slaneyMel"
																	"weighting", "warping"								// default: "warping"
																	);

/////////// CONNECTING THE ALGORITHMS ////////////////
cout << "-------- connecting algos ---------" << endl;

// Audio -> FrameCutter
std::vector<Real> audioBuffer;

audio->output("audio").set(audioBuffer);
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
cout << "-------- start processing " << audioFilename << " --------" << endl;

audio->compute();

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

  pool.add("blub", spectrum);
	
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

	//outputFile << endl;

	//printf("Max: %1.8f\n", *max_element(spectrum.begin(), spectrum.end())*100);
	counter++;
}

vector<float> vSpectrumNormalized;

vector<vector<float>> mMfccCoeffsEnlarged(mSpectrum.size(),vector<float>(mSpectrum[0].size(),0));
vector<float> vMfccCoeffsNormalized;

vector<vector<float>> mMfccBandsEnlarged(mSpectrum.size(),vector<float>(mSpectrum[0].size(),0));
vector<float> vMfccBandsNormalized;

unsigned int imageHeight;
unsigned int imageWidth;


helper::matrix_to_normalized_vector(mSpectrum, imageHeight, imageWidth, vSpectrumNormalized);
render::vector_to_PNG(audioFilename, "_spec", imageHeight, imageWidth, vSpectrumNormalized);

helper::matrix_enlarge(mMfccBands, mMfccBandsEnlarged);
helper::matrix_to_normalized_vector(mMfccBandsEnlarged, imageHeight, imageWidth, vMfccBandsNormalized);
render::vector_to_PNG(audioFilename, "_bands", imageHeight, imageWidth, vMfccBandsNormalized);
render::matrix_to_PGM(mMfccBandsEnlarged);

helper::matrix_enlarge(mMfccCoeffs, mMfccCoeffsEnlarged);
helper::matrix_to_normalized_vector(mMfccCoeffsEnlarged, imageHeight, imageWidth, vMfccCoeffsNormalized);
render::vector_to_PNG(audioFilename, "_mfcc", imageHeight, imageWidth, vMfccCoeffsNormalized);


cout << "Audio Buffer Length: " << audioBuffer.size() << endl;


delete audio;
delete fc;
delete w;
delete spec;
delete mfcc;
//delete output;

essentia::shutdown();

return 0;
}
