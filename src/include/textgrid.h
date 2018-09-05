/**
 * @file		textgrid.h
 * @class		TextGrid
 *
 * @brief		TextGrid file manipulation
 *
 *					This class meant to work with TextGrid files,
 *					which are part of the TIMIT Speach Corpus.
 *					This class reads the TextGrid file and can
 *					print the content of the class structure.
 *
 * @note		TextGrid files should all have the same structure
 *					including empty lines.
 *
 * @author	Markus Wende
 * @version 1.0
 * @date		2017-2018
 * @bug			No known bugs.
 */

#ifndef TEXTGRID_H
#define TEXTGRID_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>
#include <algorithm>
#include <string.h>
#include <thread>
#include <chrono>
#include <stdlib.h>

///	Time segment container, stores t min and max and value
typedef	struct INTERVAL_CONTAINER
{
	float xmin;
	float xmax;
	std::string text;
} interval_c;

///	Item container, whether its phonetic, vocal/consonant etc
typedef	struct ITEM_CONTAINER
{
	std::string cla;
	std::string name;
	float xmin;
	float xmax;
	int size;

	std::vector<interval_c> interval;
} item_c;

///	Root container
typedef	struct TEXTGRID_CONTAINER
{
	std::string FileType;
	std::string objectClass;
	float xmin;
	float xmax;
	bool tiers;
	int size;

	std::vector<item_c> item;
} textgrid_c;


class Textgrid
{

private:
	///	internal TextGrid container
	textgrid_c tg;

public:
	/**
	 * Constructor
	 * @param	fname contains the path and filename of the TextGrid file
	 */
	Textgrid(const char *fname);

	/// Destructor
	~Textgrid();
	
	/**	
	 * print textgrid structure
	 * @return	void
	 */
	void print_textgrid_struct();
	
	/**	
	 * GETTER
	 * get the specified item from the textGrid container 
	 * @param	num contains the item number
	 * @return	item_c structure
	 */
	item_c get_item(int num) {return tg.item[num];};
};																											// end of class TextGrid
#endif		// TEXTGRID_H
