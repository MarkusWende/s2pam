#ifndef TEXTGRID_H
#define TEXTGRID_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>
#include <algorithm>
#include <string.h>
#include <thread>										// std::this_thread::sleep_for
#include <chrono>										// std::chrono::seconds
#include <stdlib.h>									// malloc, free, rand

//-------------------------------------------------------------------------//
//               TextGrid stuff                                            //
//-------------------------------------------------------------------------//

typedef	struct INTERVAL_CONTAINER {
		float xmin;
		float xmax;
		std::string text;
} interval_c;

typedef	struct ITEM_CONTAINER {
		std::string cla;
		std::string name;
		float xmin;
		float xmax;
		int size;

		std::vector<interval_c> interval;
} item_c;

typedef	struct TEXTGRID_CONTAINER {
		std::string FileType;
		std::string objectClass;
		float xmin;
		float xmax;
		bool tiers;
		int size;

		std::vector<item_c> item;
} textgrid_c;

//----------------------------------------------------------------------------

class Textgrid {

private:
    textgrid_c tg;

public:
    Textgrid(const char *fname);
    ~Textgrid();
		
		void print_textgrid_struct();
};

//----------------------------------------------------------------------------

#endif
