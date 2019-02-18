#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

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

void apply_sift(const Mat& rgb1, const Mat& rgb2, vector<Point2f>& coord1, 
				vector<Point2f>& coord2, const Mat& mask1, const Mat& mask2, bool mask){
	const int features_count = 500;
	Ptr<xfeatures2d::SIFT> sift_detector = xfeatures2d::SIFT::create(features_count);
	Ptr<DescriptorMatcher> sift_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

	Mat final_image;
	vector<DMatch> matches;
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;
	
	if(mask==true){
		sift_detector->detectAndCompute(rgb1, mask1, keypoints1, descriptors1);
		sift_detector->detectAndCompute(rgb2, mask2, keypoints2, descriptors2);
	}
	else{
		sift_detector->detectAndCompute(rgb1, noArray(), keypoints1, descriptors1);
		sift_detector->detectAndCompute(rgb2, noArray(), keypoints2, descriptors2);	
	}

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
	save_image(final_image, "sift_image");
}

void rescale_image(const Mat& in_image, Mat& out_image, float scale){
	resize(in_image, out_image, Size(in_image.cols/scale, in_image.rows/scale));
}

void add_original(Mat& final_image, const Mat& image, const float alpha=1){
	Vec3b black_pixel{0, 0, 0};
	for(int y=0; y<image.rows; ++y){
		for(int x=0; x<image.cols; ++x){
			if(final_image.at<Vec3b>(y, x) != black_pixel){
				final_image.at<Vec3b>(y, x) = (1-alpha)*final_image.at<Vec3b>(y, x)
				+ alpha*image.at<Vec3b>(y, x);
			}
			else{
				final_image.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
			}
		}
	}
}

void get_mask(Mat& mask, int start_x, int start_y, int width, int height){
	Rect ROI(start_x, start_y, width, height);
	mask(ROI) = 255;	
}

int main(int argc, char const *argv[]){
	if(argc != 4){
		fprintf(stdout, "Usage: %s rgb1.png rgb2.png mask(1/0)\n", argv[0]);
		return 1;
	}

	Mat rgb1 = imread(argv[1], IMREAD_COLOR );
	Mat rgb2 = imread(argv[2], IMREAD_COLOR );
	if(rgb1.empty() || rgb2.empty()){
		fprintf(stdout, "Unable to open images\n");
		return 1;
	}

	const float scale = 1.4;
	rescale_image(rgb1, rgb1, scale);
	rescale_image(rgb2, rgb2, scale);

	Mat mask1(rgb1.size(), CV_8UC1, Scalar::all(0)); 
	Mat mask2(rgb2.size(), CV_8UC1, Scalar::all(0));
	get_mask(mask1, rgb1.cols/2, 0, rgb1.cols/2, rgb1.rows);
	get_mask(mask2, 0, 0, rgb2.cols/2, rgb2.rows);
	
	vector<Point2f> coordinates1;
	vector<Point2f> coordinates2;
	bool mask_activate = bool(stoi(argv[3]));
	apply_sift(rgb1, rgb2, coordinates1, coordinates2, mask1, mask2, mask_activate);

	Mat homography = findHomography(coordinates2, coordinates1, RANSAC);
	cout << "homography matrix:\n" << homography << endl;

	Mat warped_rgb;
	cv::Size warped_image_size(rgb1.cols*2, rgb1.rows*2);
	warpPerspective(rgb2, warped_rgb, homography, warped_image_size);
	display_image(warped_rgb);

	float alpha = 0.5;
	add_original(warped_rgb, rgb1, alpha);
	display_image(warped_rgb);
	save_image(warped_rgb, "panaroma");

	return 0;
}