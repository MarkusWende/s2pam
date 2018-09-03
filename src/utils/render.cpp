/**
 * @file	render.cpp
 *
 * @brief	Collection of render functions mostly for generating files from matrices
 *
 *			This namespace contains functions mostly for file generation, e.g.
 *			png's from matrices or Mel Frequency Cepstral Coefficients storage files.
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "render.h"

using namespace std;

namespace render {
void matrix_to_PGM(std::vector<std::vector<float>> m)
{
	///	construct ofstream object and initialize filename
	ofstream outputFile;
	outputFile.open("spectro.pgm");

	/// get matrix dimensions
	unsigned int height = m.size();
	unsigned int width = m[0].size();

	///	write matrix values to file
	for (int i = 1; i < height; i++) {
		for (int j = 1; j < width; j++) {
			outputFile << round(m[i][j]*255) << ' ';
		}
		outputFile << endl;
	}

	///	close file
	outputFile.close();
}

void matrix_to_MFCC_file(std::vector<std::vector<float>> m, string audioFilename)
{
	/// get filename from input path, removing all slashes and parent foldernames
	size_t found = audioFilename.find_last_of("/\\");
	string filename = audioFilename.substr(found+1);
	found = filename.find_last_of(".");
	filename = filename.substr(0,found);

	///	append the file extension .mfcc to the filename and the parent folder the file is stored in
	filename.append(".mfcc");
	filename.insert(0,"./data/FEATURES/");

	///	construct ofstream object and initialze filename
	ofstream outputFile;
	outputFile.open(filename);

	/// get matrix dimensions
	unsigned int height = m.size();
	unsigned int width = m[0].size();

	///	write matrix values to file
	for (int i = 0; i < height; i++) {
		for (int j = 1; j < width; j++) {
			outputFile << m[i][j] << ' ';
		}
		outputFile << endl;
	}

	///	close file
	outputFile.close();
}

inline void set_RGB(png_byte *ptr, float val)
{
	/// set rgb range to 255 = 8 bit
	int range = 255;
	//if (val < 0.001) val = 0.0;
	
	///	weight the input value and calculate the color in the rage 0 to 255
	int v = (int)(tanh(20*val) * range);
	if (v < 0) v = 0;
	if (v > range) v = range;

	//cout << "Val: " << val << "\tv: "<< v << endl;

	///	in dependency of the calculated color value the input value is represented by one color of
	///	6 different color areas
	if (v<=40) {
		ptr[0] = 0; ptr[1] = 0; ptr[2] = v;
	}
	else if (v>40 && v<=80) {
		ptr[0] = 0; ptr[1] = round(v/2); ptr[2] = v;
	}
	else if (v>80 && v<=140) {
		ptr[0] = 0; ptr[1] = v; ptr[2] = v;
	}
	else if (v>140 && v<=220) {
		ptr[0] = round(v/2); ptr[1] = v; ptr[2] = 0;
	}
	else if (v>220 && v<=240) {
		ptr[0] = 255-v; ptr[1] = v; ptr[2] = 0;
	}
	else if (v>240) {
		ptr[0] = v; ptr[1] = v; ptr[2] = 0;
	}
}

void vector_to_PNG(string path, string addStr, int unsigned height, int unsigned width, vector<float> v)
{
	/// get filename from input path, removing all slashes and parent foldernames
	size_t found = path.find_last_of("/\\");
	string filename = path.substr(found+1);
	found = filename.find_last_of(".");
	filename = filename.substr(0,found);
	
	///	append the additional string to the filename
	filename.append(addStr);
	
	///	append the file extension .mfcc to the filename and the parent folder the file is stored in
	filename.append(".png");
	filename.insert(0,"./data/SPECS/");

	//printf("Filename: %s\n", filename.c_str());
	//printf("Image height in px: %d\n", height);
	//printf("Image width in px: %d\n", width);

	//printf("Vector Length: %d\n", v.size());

	///	initialze file, metadata and data pointer
	FILE *fp = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;

	/// Open file for writing (binary mode)
	fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		fprintf(stderr, "Could not open file %s for writing\n", filename);
	}

	/// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Could not allocate write struct\n");
	}

	/// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Could not allocate info struct\n");
	}

	/// Setup Exception handling
	if (setjmp(png_jmpbuf(png_ptr))) {
		fprintf(stderr, "Error during png creation\n");
	}

	///	initialize input/output for the PNG file
	png_init_io(png_ptr, fp);

	/// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, width, height,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	/// Set title
	char title[100] = "Blub";
	char titleKey[10] = "Title";
	png_text title_text;
	title_text.compression = PNG_TEXT_COMPRESSION_NONE;
	title_text.key = titleKey;
	title_text.text = title;
	png_set_text(png_ptr, info_ptr, &title_text, 1);

	///	write meta data to the file
	png_write_info(png_ptr, info_ptr);

	/// Allocate memory for one row (3 bytes per pixel - RGB)
	row = (png_bytep) malloc(3 * width * sizeof(png_byte));

	/// Write image data
	int x, y;
	float max_value = 0;
	for (y=height-1 ; y>=0 ; y--) {
		for (x=0 ; x<width ; x++) {
			if(v.at(y*width + x) > max_value) {max_value = v.at(y*width + x); }
			set_RGB(&(row[x*3]), v.at(y*width + x));
		}
		png_write_row(png_ptr, row);
	}

	/// End write
	png_write_end(png_ptr, NULL);

	///	close file and free memory
	fclose(fp);
	png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	free(row);

}
}
