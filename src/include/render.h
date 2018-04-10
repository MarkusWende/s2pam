#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <malloc.h>
#include <png.h>

namespace render {
	void matrix_to_PGM(std::vector<std::vector<float>> m);
	inline void set_RGB(png_byte *ptr, float val);
	void vector_to_PNG(std::string path, std::string addStr, int unsigned height, int unsigned width, std::vector<float> v);
}
