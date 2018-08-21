/**
 * @file		textgrid.cpp
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

#include "textgrid.h"

using namespace std;

Textgrid::Textgrid(const char *fname)
{
	///	file stream declaration
	ifstream inputTextGridFile;
	inputTextGridFile.open(fname);
	
	///	initialization of the control variables
	int i = 0;
	int itemNumber = -1;
	int itemHeadCounter = -1;
	int intervalNumber = -1;
	int intervalHeadCounter = -1;

	///	safe input line from file as a string
	string line;

	///	loop over the lines in a file
	while (getline(inputTextGridFile, line))
	{
		/**
		 * increase line counter i by 1
		 * assign line string to a temp stringt str and
		 * remove all white spaces
		 */
		i++;
		string str;
		str.assign(line.begin(),line.end());
		str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
		
		///	find lines with the char =
		size_t found = str.find_last_of("=");
		string sub = str.substr(found+1);

		/**
		 * file header
		 * if file line 1: read file type (e.g File type = "ooTextFile")
		 * if line 2:	read object class (e.g Object class = "TextGrid")
		 * quotation marks are romoved
		 *
		 * if line 4: read audio file start in seconds (e.g xmin = 0)
		 * if line 5: read audio file end in seconds (e.g xmax = 3.968375)
		 * if line 7: read how many items the file have (e.g size = 5)
		 */
		if (i == 1)
		{
			sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
			tg.FileType.assign(sub);
		}
		if (i == 2)
		{
			sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
			tg.objectClass.assign(sub);
		}
		if (i == 4)
			tg.xmin = stof(sub);
		if (i == 5)
			tg.xmax = stof(sub);
		if (i == 7)
			tg.size = stoi(sub);

		///	find lines with the string item
		found = line.find("item");
		
		/**
		 * file items
		 * if the string item was found, the item number is extracted by searching for the char [
		 * if the string that was found is a digit, the number is assigned to itemNumber
		 * a new item instance of the struct item_c is allocated
		 * and the itemHeadCounter variable is set to the begin of the head
		 */
		if (found!=string::npos)
		{
			found = str.find_last_of("[");
			string sub = str.substr(found+1);
			found = sub.find_last_of("]");
			sub = sub.substr(0,found);
			if (isdigit(sub[0]))
			{
				itemNumber = stoi(sub) - 1;
				tg.item.push_back(item_c());
				itemHeadCounter = 0;
			}
		}

		/// find lines with the string intervals
		found = line.find("intervals");

		/**
		 * file intervals
		 * if the string intervals was found, the interval number is extracted by searching for the char [
		 * in the string line and if the result is a digit the number is assigned to intervalNumber
		 * a new interval instance of the struct interval_c is allocated and the intervalHeadCounter
		 * is set to the begin of the interval head
		 */
		if (found!=string::npos)
		{
			found = str.find_last_of("[");
			string sub = str.substr(found+1);
			found = sub.find_last_of("]");
			sub = sub.substr(0,found);
			if (isdigit(sub[0]))
			{
				intervalNumber = stoi(sub) - 1;
				tg.item[itemNumber].interval.push_back(interval_c());
				intervalHeadCounter = 0;
			}
		}

		///	item head, interval head and data read
		if (itemNumber>-1)
		{
			/// search for strings with the char =
			found = str.find_last_of("=");
			
			/// save located string with the char = to a sub string for further manipulation
			string sub = str.substr(found+1);

			/**
			 * item head
			 * read item head an store information into the container
			 * if itemHeadCounter 1: read item class (e.g class = "IntervalTier")
			 * if itemHeadCounter 2: read item name (e.g name = "phn")
			 * quotation marks are romoved
			 * 
			 * if itemHeadCounter 3: read item start in seconds (e.g xmin = 0)
			 * if itemHeadCounter 4: read item end in seconds (e.g xmax = 3.968375)
			 * if itemHeadCounter 5: read interval size (e.g size = 41) and set the itemHeadCounter back to -1
			 * also increase the itemHeadCounter every time until hea reaches 5
			 */
			if (itemHeadCounter == 1)
			{
				sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
				tg.item[itemNumber].cla.assign(sub);
			}
			if (itemHeadCounter == 2)
			{
				sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
				tg.item[itemNumber].name.assign(sub);
			}
			if (itemHeadCounter == 3)
				tg.item[itemNumber].xmin = stof(sub);
			if (itemHeadCounter == 4)
				tg.item[itemNumber].xmax = stof(sub);
			if (itemHeadCounter == 5) {
				tg.item[itemNumber].size = stoi(sub);
				itemHeadCounter = -1;
			}
			if (itemHeadCounter >= 0)
				itemHeadCounter++;

			/**
			 * interval head and data
			 * read interval head and file data
			 * if intervalHeadCounter 1: read interval start in seconds (e.g xmin = 0.7025)
			 * if intervalHeadCounter 2: read interval end in seconds (e.g xmax = 0.7989375)
			 * if intervalHeadCounter 3: read interval text (e.g text = "iy")
			 * increase the intervalHeadCounter every time by 1 if he is >= 0
			 */
			if (itemHeadCounter == -1)
			{
				if (intervalHeadCounter == 1)
					tg.item[itemNumber].interval[intervalNumber].xmin = stof(sub);
				if (intervalHeadCounter == 2)
					tg.item[itemNumber].interval[intervalNumber].xmax = stof(sub);
				if (intervalHeadCounter == 3) {
					sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
					tg.item[itemNumber].interval[intervalNumber].text.assign(sub);
				}
				if (intervalHeadCounter >= 0)
					intervalHeadCounter++;
			}
		}

	}		/// while
}			/// TextGrid

Textgrid::~Textgrid() {

}

void Textgrid::print_textgrid_struct() {
	///	print file head to console
	printf("File type = \"%s\"\n", tg.FileType.c_str());
	printf("Object class = \"%s\"\n\n", tg.objectClass.c_str());
	printf("xmin = %f\n", tg.xmin);
	printf("xmax = %f\n", tg.xmax);
	printf("tiers? <exists>\n");
	printf("size = %d\n", tg.size);
	printf("item []:\n");

	/// loop over textgrid items
	for (int i = 0; i < tg.size; i++) {
		/// print item head to console
		printf("\titem [%d]:\n", i+1);
		printf("\t\tclass = \"%s\"\n", tg.item[i].cla.c_str());
		printf("\t\tname = \"%s\"\n", tg.item[i].name.c_str());
		printf("\t\txmin = %f\n", tg.item[i].xmin);
		printf("\t\txmax = %f\n", tg.item[i].xmax);
		printf("\t\tintervals: size = %d\n", tg.item[i].size);

		/// loop over textgrid intervals
		for (int j = 0; j < tg.item[i].size; j++) {
			/// print interval head and data to console
			printf("\t\tintervals [%d]:\n", j+1);
			printf("\t\t\txmin = %f\n", tg.item[i].interval[j].xmin);
			printf("\t\t\txmax = %f\n", tg.item[i].interval[j].xmax);
			printf("\t\t\ttext = \"%s\"\n", tg.item[i].interval[j].text.c_str());
		}
	}
}
