#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct Parameters{
	int num_images;
	int vertical_corners;
	int horizontal_corners;
	int total_points;
	Size board_size;

	Parameters(int imgs=8, int ver=6, int hor=4):num_images{imgs},
	vertical_corners{ver}, horizontal_corners{hor}{
		total_points = vertical_corners * horizontal_corners;
		board_size = Size(horizontal_corners, vertical_corners);
	};
};

void parse_images(const char* file, vector<string>& images_path){
	ifstream file_read(file);
	string line;
	if(file_read.is_open()){
		while(getline(file_read, line)){
			images_path.push_back(line);
		}
	}
}

void generate_world_points(vector<vector<Point3f>>& points, const Parameters& param){
	vector<Point3f> single_board;
	for(int i=0; i<param.horizontal_corners; ++i){
		for(int j=0; j<param.vertical_corners; ++j){
			single_board.push_back(Point3f(i, j, 0));	
		}
	}

	for(int i=0; i<param.num_images; ++i){
		points.push_back(single_board);
	}
}

void display_image(const Mat& image){
	namedWindow("opencv_viewer", WINDOW_AUTOSIZE);
	imshow("opencv_viewer", image);
	waitKey(0);
	destroyWindow("opencv_viewer");
}

void preprocess_image(const Mat& in_image, Mat& out_image){
	resize(in_image, out_image, Size(640, 480));
}

int main(int argc, char const *argv[]){
	if(argc != 2){
		fprintf(stdout, "Usage: %s images_file.txt\n", argv[0]);
	}

	vector<string> images_path;
	parse_images(argv[1], images_path);
	
	Parameters param{};

	vector<vector<Point3f>> object_points;
	vector<vector<Point2f>> image_points;
	vector<Point2f> detected_points;

	generate_world_points(object_points, param);

	for(int i=0; i<param.num_images; ++i){
		Mat rgb = imread(images_path[i], IMREAD_COLOR);
		
		Mat3b processed_image;
		preprocess_image(rgb, processed_image);

		Mat gray_image;
		cvtColor(processed_image, gray_image, COLOR_BGR2GRAY);

		vector<Point2f> corners;
		bool found = findChessboardCorners(processed_image, param.board_size, corners);
		
		if(found){
			cout << "Inside found: " << i << endl;
			cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1),
						 TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));
			drawChessboardCorners(gray_image, param.board_size, corners, found);
		}

		display_image(processed_image);
		display_image(gray_image);
	}

	return 0;
}