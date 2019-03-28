#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void display_image(const Mat& image){
	namedWindow("opencv_viewer", WINDOW_AUTOSIZE);
	imshow("opencv_viewer", image);
	waitKey(0);
	destroyWindow("opencv_viewer");
}

void save_image(const Mat& image, string image_name){
	image_name += ".jpg";
	imwrite(image_name, image);
	cout << image_name << " saved in current directory.\n";
}

void rescale_image(const Mat& in_image, Mat& out_image, float scale){
	resize(in_image, out_image, Size(in_image.cols/scale, in_image.rows/scale));
}

void split_image(const Mat3b& rgb, Mat3b& new_rgb, int offset){
	for(int y=0; y<new_rgb.rows; ++y){
		for(int x=0; x<new_rgb.cols; ++x){
			new_rgb.at<Vec3b>(y, x) = rgb.at<Vec3b>(y, x+offset);
		}
	}
}

void apply_sift(const Mat& rgb1, const Mat& rgb2){
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;
	const int features_count = 500000;
	Ptr<xfeatures2d::SIFT> feature_detector = xfeatures2d::SIFT::create(features_count);
	feature_detector->detectAndCompute(rgb1, noArray(), keypoints1, descriptors1);
	feature_detector->detectAndCompute(rgb2, noArray(), keypoints2, descriptors2);	

	vector<vector<DMatch>> knn_matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	
	vector<DMatch> matches;
	const float threshold_ratio = 0.7f;
	for(size_t i=0; i<knn_matches.size(); ++i){
		if(knn_matches[i][0].distance < threshold_ratio * knn_matches[i][1].distance)
			matches.push_back(knn_matches[i][0]);
	}
	
	Mat final_image;
	drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, final_image);
	save_image(final_image, "dense_sift");
	display_image(final_image);
}

int main(int argc, char const *argv[]){
	if(argc != 2){
		fprintf(stdout, "Usage: %s rgb.png\n", argv[0]);
		return 1;
	}
	Mat rgb = imread(argv[1], IMREAD_COLOR );
	if(rgb.empty()){
		fprintf(stdout, "Unable to open image\n");
		return 1;
	}

	const float scale = 1.1;
	rescale_image(rgb, rgb, scale);

	Size new_size = Size(rgb.rows, rgb.cols/2);
	Mat3b rgb1(new_size, CV_8UC3);
	Mat3b rgb2(new_size, CV_8UC3);
	split_image(rgb, rgb1, 0);
	split_image(rgb, rgb2, rgb.cols/2);

	display_image(rgb);
	display_image(rgb1);
	display_image(rgb2);
	
	apply_sift(rgb1, rgb2);

	return 0;
}
