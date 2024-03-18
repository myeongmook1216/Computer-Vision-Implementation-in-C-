#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

#define PI 3.141592653589793238462643383279502884L

using namespace cv;
using namespace std;

Mat Gaussian_Kernel(int , double);
Mat img_draw_func_double(Mat& );
Mat img_draw_func_uchar(Mat&);
Mat add_Gaussian_Noise(Mat&,double, double);
Mat convolution_Gaussian(Mat&, Mat&);

int main() {
	Mat img_gray = imread("lena.jpg", IMREAD_GRAYSCALE);
	Mat img_crop = imread("img_crop.jpg", IMREAD_GRAYSCALE);
	Mat img_crop_draw = img_draw_func_uchar(img_crop);
	Mat img_kernel = Gaussian_Kernel(5, 0.7);
	Mat img_kernel_draw = img_draw_func_double(img_kernel);

	Mat img_gray_GN = add_Gaussian_Noise(img_gray, 0, 30);
	Mat img_crop_GN = add_Gaussian_Noise(img_crop, 0, 10);
	Mat img_gray_GN_10 = add_Gaussian_Noise(img_gray, 0, 10);
	Mat img_gray_GN_20 = add_Gaussian_Noise(img_gray, 0, 20);
	Mat img_gray_GN_30 = add_Gaussian_Noise(img_gray, 0, 30);
	Mat img_crop_GN_draw = img_draw_func_uchar(img_crop_GN);
	Mat img_conv_G = convolution_Gaussian(img_gray_GN, img_kernel);
	Mat img_conv_crop_G = convolution_Gaussian(img_crop_GN, img_kernel);
	
	Mat img_conv_G_draw = img_draw_func_uchar(img_conv_G);
	Mat img_conv_crop_G_draw = img_draw_func_uchar(img_conv_crop_G);

	namedWindow("asd", WINDOW_AUTOSIZE);
	/*
	imshow("asd1", img_conv_G);
	imshow("asd2", img_gray_GN);
	imshow("asd5", img_kernel_draw);
	imshow("asd4", img_crop_GN_draw);
	imshow("asd3", img_conv_crop_G_draw);
	imshow("img_gray", img_gray);
	imshow("add_gaussain_noise_10", img_gray_GN_10);
	imshow("add_gaussain_noise_20", img_gray_GN_20);
	imshow("add_gaussain_noise_30", img_gray_GN_30);
	*/
	imshow("img_gray_GN", img_gray_GN);
	imshow("img_conv_G", img_conv_G);
	waitKey();
	
	}

Mat convolution_Gaussian(Mat& img, Mat& kernel) {
	int kernelCenter = (kernel.rows - 1) / 2;
	Mat img_conv(img.cols - 2 * kernelCenter, img.rows - 2 * kernelCenter, CV_8UC1);

	for (int i = kernelCenter; i < img.rows - kernelCenter; i++) {
		for (int j = kernelCenter; j < img.cols - kernelCenter; j++) {
			double value = 0;
			for (int m = 0; m < kernel.rows; m++) {
				for (int n = 0; n < kernel.cols; n++) {
					value += static_cast<double>(img.at<uchar>(i - kernelCenter + m, 
						j - kernelCenter + n)) * kernel.at<double>(m, n);
				}
			}

			img_conv.at<uchar>(i - kernelCenter, j - kernelCenter) = static_cast<uchar>(value);
		}
	}
	
	return img_conv;
}


Mat add_Gaussian_Noise(Mat& img, double mean, double std) {
	Mat noise(img.size(), CV_8UC1); 
	Mat result(img.size(), img.type()); 

	unsigned seed = static_cast<unsigned>(chrono::system_clock::now().time_since_epoch().count());
	default_random_engine generator(seed);
	normal_distribution<double> distribution(mean, std);

	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			double noise_val = distribution(generator);
			result.at<uchar>(y, x) = saturate_cast<uchar>(img.at<uchar>(y, x) + noise_val);
		}
	}

	return result;
}


Mat Gaussian_Kernel(int size, double sigma) { 
	Mat kernel(size, size, CV_64F);

	int kernelCenter = (size - 1) / 2;
	double sum = 0.0;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int x = kernelCenter - i;
			int y = kernelCenter - j;
			kernel.at<double>(i, j) = exp(-(static_cast<double>(x * x) +
				static_cast<double>(y * y)) / (2.0 * sigma * sigma)) / sqrt(2.0 * PI * sigma * sigma);
			sum += kernel.at<double>(i, j);
		}
	}
	kernel /= sum;
	return kernel;
}



Mat img_draw_func_double(Mat& img) { 
	double scale = 512 / img.rows;   // 행과 열의 scale 같다고 가정	

	Mat img_draw(img.cols * scale, img.rows * scale, CV_64FC1, Scalar(0, 0, 0));
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			string text = to_string(img.at<double>(i, j));
			int text_width, text_height;

			Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
			text_width = text_size.width;
			text_height = text_size.height;
			int x = j * scale + (scale - text_width) / 2;
			int y = i * scale + (scale + text_height) / 2;

			putText(img_draw, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
		}

	return img_draw;
}


Mat img_draw_func_uchar(Mat& img) { 
	double scale = 512 / img.rows;   // 행과 열의 scale 같다고 가정	

	Mat img_draw(img.cols * scale, img.rows * scale, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			string text = to_string(static_cast<int>(img.at<uchar>(i, j)));
			int text_width, text_height;

			Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
			text_width = text_size.width;
			text_height = text_size.height;
			int x = j * scale + (scale - text_width) / 2;
			int y = i * scale + (scale + text_height) / 2;

			putText(img_draw, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
		}

	return img_draw;
}
