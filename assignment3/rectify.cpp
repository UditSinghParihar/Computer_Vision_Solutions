#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "epilines.h"

using namespace std;
using namespace cv;

void display_image(const Mat& image){
	namedWindow("opencv_viewer", WINDOW_AUTOSIZE);
	imshow("opencv_viewer", image);
	waitKey(0);
	destroyWindow("opencv_viewer");
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

void apply_sift(const Mat& rgb1, const Mat& rgb2, vector<Point2f>& coord1, 
				vector<Point2f>& coord2){
	const int features_count = 500;
	Ptr<xfeatures2d::SIFT> sift_detector = xfeatures2d::SIFT::create(features_count);
	Ptr<DescriptorMatcher> sift_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

	Mat final_image;
	vector<DMatch> matches;
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;
	
	sift_detector->detectAndCompute(rgb1, noArray(), keypoints1, descriptors1);
	sift_detector->detectAndCompute(rgb2, noArray(), keypoints2, descriptors2);	

	vector<vector<DMatch>> knn_matches;
	sift_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	
	const float threshold_ratio = 0.7f;
	for(size_t i=0; i<knn_matches.size(); ++i){
		if(knn_matches[i][0].distance < threshold_ratio * knn_matches[i][1].distance)
			matches.push_back(knn_matches[i][0]);
	}
	drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, final_image);
	
	for(int i=0; i<matches.size(); ++i){
		coord1.push_back(keypoints1[matches[i].queryIdx].pt);
		coord2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
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

	const float scale = 1.7;
	rescale_image(rgb, rgb, scale);

	Size new_size = Size(rgb.rows, rgb.cols/2);
	Mat3b rgb1(new_size, CV_8UC3);
	Mat3b rgb2(new_size, CV_8UC3);
	split_image(rgb, rgb1, 0);
	split_image(rgb, rgb2, rgb.cols/2);

	display_image(rgb);
	display_image(rgb1);
	display_image(rgb2);
	
	vector<Point2f> coord1;
	vector<Point2f> coord2;
	apply_sift(rgb1, rgb2, coord1, coord2);

	Mat F, H1, H2;
	F = findFundamentalMat(coord1, coord2, CV_FM_RANSAC);
	stereoRectifyUncalibrated(coord1, coord2, F, new_size, H1, H2);
	
	Mat warped_rgb1, warped_rgb2;
	cv::Size warped_image_size(rgb1.cols*2, rgb1.rows);
	warpPerspective(rgb1, warped_rgb1, H1, warped_image_size);
	warpPerspective(rgb2, warped_rgb2, H2, warped_image_size);
	
	display_image(warped_rgb1);
	display_image(warped_rgb2);

	vector<Point2f> p1;
	vector<Point2f> p2;
	apply_sift(warped_rgb1, warped_rgb2, p1, p2);
	vector<Point2f> p1_few{p1[0], p1[1]};
	vector<Point2f> p2_few{p2[0], p2[1]};
	drawEpipolarLines<float, float>("Epiplines", F, warped_rgb1, warped_rgb2, p1_few, p2_few);
	
	return 0;
}