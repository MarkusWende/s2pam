#include "textgrid.h"

using namespace std;

Textgrid::Textgrid(const char *fname) {
	ifstream inputTextGridFile;
	inputTextGridFile.open(fname);
	
	int i = 0;
	int itemNumber = -1;
	int itemHeadCounter = -1;
	int intervalNumber = -1;
	int intervalHeadCounter = -1;
	string line;
	while (getline(inputTextGridFile, line)) {
	
		i++;
		string str;
		str.assign(line.begin(),line.end());
		str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
		
		
		size_t found = str.find_last_of("=");
		string sub = str.substr(found+1);
		if (i == 1) {
			sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
			tg.FileType.assign(sub);
		}
		if (i == 2) {
			sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
			tg.objectClass.assign(sub);
		}
		if (i == 4) {
			tg.xmin = stof(sub);
		}
		if (i == 5) {
			tg.xmax = stof(sub);
		}
		if (i == 7) {
			tg.size = stoi(sub);
		}


		found = line.find("item");
		if (found!=string::npos) {
			found = str.find_last_of("[");
			string sub = str.substr(found+1);
			found = sub.find_last_of("]");
			sub = sub.substr(0,found);
			if (isdigit(sub[0])) {
				itemNumber = stoi(sub) - 1;
				tg.item.push_back(item_c());
				itemHeadCounter = 0;
			}
		}
		found = line.find("intervals");
		if (found!=string::npos) {
			found = str.find_last_of("[");
			string sub = str.substr(found+1);
			found = sub.find_last_of("]");
			sub = sub.substr(0,found);
			if (isdigit(sub[0])) {
				intervalNumber = stoi(sub) - 1;
				tg.item[itemNumber].interval.push_back(interval_c());
				intervalHeadCounter = 0;
			}
		}
		if (itemNumber>-1) {
			found = str.find_last_of("=");
			string sub = str.substr(found+1);
			if (itemHeadCounter == 1) {
				sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
				tg.item[itemNumber].cla.assign(sub);
			}
			if (itemHeadCounter == 2) {
				sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
				tg.item[itemNumber].name.assign(sub);
			}
			if (itemHeadCounter == 3) {
				tg.item[itemNumber].xmin = stof(sub);
			}
			if (itemHeadCounter == 4) {
				tg.item[itemNumber].xmax = stof(sub);
			}
			if (itemHeadCounter == 5) {
				tg.item[itemNumber].size = stoi(sub);
				itemHeadCounter = -1;
			}
			if (itemHeadCounter >= 0)
				itemHeadCounter++;
			if (itemHeadCounter == -1) {
				if (intervalHeadCounter == 1) {	
					tg.item[itemNumber].interval[intervalNumber].xmin = stof(sub);
				}
				if (intervalHeadCounter == 2) {	
					tg.item[itemNumber].interval[intervalNumber].xmax = stof(sub);
				}
				if (intervalHeadCounter == 3) {	
					sub.erase(remove(sub.begin(), sub.end(), '"'), sub.end());
					tg.item[itemNumber].interval[intervalNumber].text.assign(sub);
				}
				if (intervalHeadCounter >= 0)
					intervalHeadCounter++;
			}
		}

	}
}

Textgrid::~Textgrid() {

}

void Textgrid::print_textgrid_struct() {
	printf("File type = \"%s\"\n", tg.FileType.c_str());
	printf("Object class = \"%s\"\n\n", tg.objectClass.c_str());
	printf("xmin = %f\n", tg.xmin);
	printf("xmax = %f\n", tg.xmax);
	printf("tiers? <exists>\n");
	printf("size = %d\n", tg.size);
	printf("item []:\n");

	for (int i = 0; i < tg.size; i++) {
		printf("\titem [%d]:\n", i+1);
		printf("\t\tclass = \"%s\"\n", tg.item[i].cla.c_str());
		printf("\t\tname = \"%s\"\n", tg.item[i].name.c_str());
		printf("\t\txmin = %f\n", tg.item[i].xmin);
		printf("\t\txmax = %f\n", tg.item[i].xmax);
		printf("\t\tintervals: size = %d\n", tg.item[i].size);

		for (int j = 0; j < tg.item[i].size; j++) {
			printf("\t\tintervals [%d]:\n", j+1);
			printf("\t\t\txmin = %f\n", tg.item[i].interval[j].xmin);
			printf("\t\t\txmax = %f\n", tg.item[i].interval[j].xmax);
			printf("\t\t\ttext = \"%s\"\n", tg.item[i].interval[j].text.c_str());
		}
	}
}
