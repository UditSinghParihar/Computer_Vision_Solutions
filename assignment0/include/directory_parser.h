#ifndef DIRECTORY_PARSER_H
#define DIRECTORY_PARSER_H

#include <iostream>
#include <dirent.h>
#include <string>
#include <algorithm>
#include <cctype>

using namespace std;

class ParseImages{
private:
	const char* rgb_folder;
	vector<string>& rgb_images;	

private:
	int get_image_number(const string &str){
		int index = 0;
		string result{};
		while(str[index] != '.'){
			result += str[index];
			++index;
		}
		return stoi(result);
	}

	void get_extension(const string& str, string& extension){
		int index = 0; 
		while(str[index] != '.'){
			++index;
		}
		while(index != str.size()){
			extension += str[index];
			++index;
		}
	}
	
	void get_image_path(int image_index, const string& extension, 
							const char* directory, string& image_path){
		int index=0;
		string path{};
		while(directory[index] != '\0'){
			path += directory[index];
			++index;
		}
		
		image_path = path + to_string(image_index) + extension;
	}

	void parse_images(vector<string>& images, const char* folder){
		string extension;
		bool extension_parsed = false;
		vector<int> list;
		DIR* directory = opendir(folder);
		dirent* pointer;
		while((pointer = readdir(directory)) != NULL){
			if(pointer->d_name[0] != '.'){
				list.push_back(get_image_number(pointer->d_name));
				if(extension_parsed == false){
					get_extension(pointer->d_name, extension);
					extension_parsed = true;
				}
			}
		}
		closedir(directory);
		sort(list.begin(), list.end());

		for(int i=0; i<list.size(); ++i){
			string image_path;
			get_image_path(list[i], extension, folder, image_path);
			images.push_back(image_path);
		}
	}

public:
	ParseImages(const char* rgb, vector<string>& rgb_list):
				rgb_folder{rgb},
				rgb_images{rgb_list}{};

	void start_processing(void){
		parse_images(rgb_images, rgb_folder);
	}
	
};

#endif