#include <iostream>
#include <string>
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
	static int index = 1;
	image_name += to_string(index) + ".jpg";
	imwrite(image_name, image);
	cout << image_name << " saved in current directory.\n";
	++index;
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
	save_image(final_image, "sift_image_taj");
}

void rescale_image(const Mat& in_image, Mat& out_image, float scale){
	resize(in_image, out_image, Size(in_image.cols/scale, in_image.rows/scale));
}

void add_original(Mat& final_image, const Mat& image, const float alpha=0.5){
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

void parse_images(const char* file, vector<string>& images_path){
	ifstream file_read(file);
	string line;
	if(file_read.is_open()){
		while(getline(file_read, line)){
			images_path.push_back(line);
		}
	}
}

int get_homography(const Mat& rgb1, const Mat& rgb2, Mat& homography){
	vector<Point2f> coordinates1;
	vector<Point2f> coordinates2;
	apply_sift(rgb1, rgb2, coordinates1, coordinates2);

	homography = findHomography(coordinates2, coordinates1, RANSAC);
	cout << "homography matrix:\n" << homography << endl;

	Mat warped_rgb;
	cv::Size warped_image_size(rgb1.cols*2, rgb1.rows*2);
	warpPerspective(rgb2, warped_rgb, homography, warped_image_size);
	display_image(warped_rgb);

	float alpha = 0.5;
	add_original(warped_rgb, rgb1, alpha);
	display_image(warped_rgb);
}

void accumulate(const Mat& warp_image, Mat& final_image){
	Vec3b black_pixel{0, 0, 0};

	for(int y=0; y<warp_image.rows; ++y){
		for(int x=0; x<warp_image.cols; ++x){
			if(warp_image.at<Vec3b>(y, x) != black_pixel){
				final_image.at<Vec3b>(y, x) = warp_image.at<Vec3b>(y, x);
			}
		}
	}
}

void get_panaroma(const vector<Mat>& images, const vector<Mat>& homographies){
	cv::Size panaroma_size(images[0].cols*2, images[0].rows*2);
	Mat panaroma(panaroma_size, images[0].type());

	for(int i=0; i<images.size()-1; ++i){
		Mat warped_rgb;
		warpPerspective(images[i+1], warped_rgb, homographies[i], panaroma_size);
		accumulate(warped_rgb, panaroma);
	}

	display_image(panaroma);
	save_image(panaroma, "panaroma_taj");
}

int main(int argc, char const *argv[]){
	if(argc != 2){
		fprintf(stdout, "Usage: %s images_file.txt\n", argv[0]);
		return 1;
	}

	vector<string> images_path;
	parse_images(argv[1], images_path);
	
	vector<Mat> images;
	for(int i=0; i<images_path.size(); ++i){
		Mat rgb = imread(images_path[i], IMREAD_COLOR);
		const float scale = 1.4;
		rescale_image(rgb, rgb, scale);

		images.push_back(rgb);
	}

	vector<Mat> homographies;
	for(int i=0; i<images.size()-1; ++i){
		Mat homography;
		get_homography(images[0], images[i+1], homography);
		homographies.push_back(homography);
	}

	get_panaroma(images, homographies);

	return 0;
}