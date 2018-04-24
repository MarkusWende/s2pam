#include "wave_read.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

Wave::Wave(const char* filePath)
{
	int headerSize = sizeof(wav_hdr), filelength = 0;

	FILE* wavFile = fopen(filePath, "r");
	if (wavFile == nullptr)
	{
		fprintf(stderr, "Unable to open wave file: %s\n", filePath);
	}

	//Read the header
  size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
  //cout << "Header Read " << bytesRead << " bytes." << endl;
	if (bytesRead > 0)
	{
		//Read the data
		uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
		uint64_t numSamples = wavHeader.ChunkSize / bytesPerSample; //How many samples are in the wav file?
		static const uint16_t BUFFER_SIZE = 8192;
		int16_t* buffer = new int16_t[BUFFER_SIZE];
		while ((bytesRead = fread(buffer, sizeof buffer[0], BUFFER_SIZE / (sizeof buffer[0]), wavFile)) > 0)
		//fread(buffer, sizeof buffer[0], BUFFER_SIZE / (sizeof buffer[0]), wavFile);
		{
				/** DO SOMETHING WITH THE WAVE DATA HERE **/
			//cout << "Read " << bytesRead << " bytes." << endl;
			/*for (int i = 0; i < bytesRead*8; i++) {
				samples.push_back(buffer[i]);
			}*/
			samples.insert(samples.end(), buffer, buffer + bytesRead);
		}
		delete [] buffer;
		buffer = nullptr;
		//filelength = getFileSize(wavFile);

	}
	fclose(wavFile);
}

Wave::~Wave()
{

}

void Wave::showHeader()
{
		cout	<< "RIFF header                :" << wavHeader.RIFF[0]		<< wavHeader.RIFF[1]
					<< wavHeader.RIFF[2]							<< wavHeader.RIFF[3]		<< endl;
		cout	<< "WAVE header                :" << wavHeader.WAVE[0]		<< wavHeader.WAVE[1]
					<< wavHeader.WAVE[2]							<< wavHeader.WAVE[3]		<< endl;
		cout	<< "FMT                        :" << wavHeader.fmt[0]			<< wavHeader.fmt[1]
					<< wavHeader.fmt[2]								<< wavHeader.fmt[3]			<< endl;
		cout	<< "File size                  :" << wavHeader.ChunkSize	<< endl;

		// Display the sampling Rate from the header
		cout	<< "Sampling Rate              :" << wavHeader.SamplesPerSec	<< endl;
		cout	<< "Number of bits used        :" << wavHeader.bitsPerSample	<< endl;
		cout	<< "Number of channels         :" << wavHeader.NumOfChan			<< endl;
		cout	<< "Number of bytes per second :" << wavHeader.bytesPerSec		<< endl;
		cout	<< "Data length                :" << wavHeader.Subchunk2Size	<< endl;
		cout	<< "Audio Format               :" << wavHeader.AudioFormat		<< endl;
		// Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM

		cout	<< "Block align                :" << wavHeader.blockAlign			<< endl;
		cout	<< "Data string                :" << wavHeader.Subchunk2ID[0]
					<< wavHeader.Subchunk2ID[1]				<< wavHeader.Subchunk2ID[2]
					<< wavHeader.Subchunk2ID[3]				<< endl;
}
