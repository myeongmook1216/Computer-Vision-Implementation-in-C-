#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void calculateHistogram(const Mat&, int*);
Mat drawHistogram(const int*);
Mat binaryThreshold(const Mat&, int);
int otsuAlgorithm(const int*);
Mat equalizeHistogram(const int*, const Mat&);

int main() {
	Mat img_gray = imread("lena.jpg", IMREAD_GRAYSCALE);
	int histogram[256]; calculateHistogram(img_gray, histogram);
	Mat draw_h = drawHistogram(histogram);
	Mat img_equalize = equalizeHistogram(histogram, img_gray);
	Mat img_gray_binary = binaryThreshold(img_gray, otsuAlgorithm(histogram));
	cout << otsuAlgorithm(histogram);
	namedWindow("histogram", WINDOW_AUTOSIZE);
	imshow("img_gray", img_gray);
	imshow("img_gray_equalize", img_equalize);
	waitKey();
}


Mat equalizeHistogram(const int* histogram, const Mat& img_input) {
	int cumulative_histogram[256] = { 0 };
	cumulative_histogram[0] = histogram[0];
	for (int i = 1; i < 256; i++) { cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]; }

	int total_pixel_value = 0;
	for (int i = 0; i < 256; i++) { total_pixel_value += histogram[i]; }
	double normalization_factor = 255.0 / total_pixel_value;

	int normalized_cumulative_histogram[256];
	for (int i = 0; i < 256; i++) { normalized_cumulative_histogram[i] = cvRound(cumulative_histogram[i] * normalization_factor); }

	Mat img_output = img_input.clone();
	for (int y = 0; y < img_output.rows; y++) {
		for (int x = 0; x < img_output.cols; x++) {
			int pixel_value = static_cast<int>(img_output.at<uchar>(y, x)); 
			img_output.at<uchar>(y, x) = static_cast<uchar>(normalized_cumulative_histogram[pixel_value]);
		}
	}

	return img_output;
}



int otsuAlgorithm(const int* histogram) {
	double total_pixel = 0;	double normalized_histogram[256] = { 0 };
	for (int i = 0; i < 256; i++) { total_pixel += histogram[i]; }
	for (int i = 0; i < 256; i++) { normalized_histogram[i] = histogram[i] / total_pixel; }
	int threshold = 0;	double sum1 = 0;  double sum2 = 0;
	double mean1 = 0; double mean2 = 0; double varMax = 0;

	for (int T = 0; T < 256; T++) {
		sum1 += normalized_histogram[T];
		sum2 = 1.0 - sum1;

		if (sum1 > 0 && sum2 > 0) {
			mean1 = 0;
			mean2 = 0;

			for (int i = 0; i <= T; i++) { mean1 += i * normalized_histogram[i]; }
			mean1 /= sum1;

			for (int i = T + 1; i < 256; i++) { mean2 += i * normalized_histogram[i]; }
			mean2 /= sum2;

			double varBetween = sum1 * sum2 * (mean1 - mean2) * (mean1 - mean2);

			if (varBetween > varMax) {
				varMax = varBetween;
				threshold = T;
			}
		}
	}
	return threshold;
}


void calculateHistogram(const Mat& img, int histogram[256]) {
	for (int i = 0; i < 256; ++i) 
		histogram[i] = 0;

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int pixel_value = static_cast<int>(img.at<uchar>(y, x));
			histogram[pixel_value]++;
		}
	}
}


Mat drawHistogram(const int* histogram) {
	Mat histogram_img(400, 512, CV_8UC3, Scalar(255, 255, 255));
	int hist_w = histogram_img.cols ;  	int hist_h = histogram_img.rows;
	int bin_w = cvRound((double)hist_w / 256);

	int max_value = 0;
	for (int i = 0; i < 256; i++) { if (histogram[i] > max_value) { max_value = histogram[i]; } }

	for (int i = 0; i < 256; i++) {
		int h = cvRound(histogram[i] * hist_h / max_value);

		line(histogram_img, Point(bin_w * i, hist_h), Point(bin_w * i, hist_h - h), Scalar(255, 0, 0), 2, 8, 0);
	}

	return histogram_img;
}


Mat binaryThreshold(const Mat& img_input, int threshold) {
	Mat img_binary = img_input.clone();
	int pixel_value = 0;

			for (int y = 0; y < img_binary.rows; y++) 	
				for (int x = 0; x < img_binary.cols; x++) {
			pixel_value = static_cast<int>(img_input.at<uchar>(y,x));
			img_binary.at<uchar>(y, x) = (pixel_value >= threshold) ? 255 : 0;
		}

	return img_binary;
}

