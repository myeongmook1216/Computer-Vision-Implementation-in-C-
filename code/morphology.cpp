#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat img_draw_func(Mat&);
Mat Morphology_dilate(Mat&, Mat&);
Mat Morphology_erode(Mat&, Mat&);

int main() {
	Mat img_gray = imread("lena.jpg", IMREAD_GRAYSCALE);
	Rect imgtocrop(50, 30, 16, 16);
	Mat img_crop = img_gray(imgtocrop).clone();
	Mat img_crop_draw = img_draw_func(img_crop);
	Mat img_binary;  threshold(img_gray, img_binary, 128, 255, cv::THRESH_BINARY);
	
	Mat img_binary_crop = img_binary(imgtocrop).clone();
	Mat img_binary_draw = img_draw_func(img_binary_crop);

	Mat kernel = Mat::ones(3, 3, CV_8U) * 255;
	
	Mat img_dilate = Morphology_dilate(img_binary_crop, kernel);
	Mat img_dilate_draw = img_draw_func(img_dilate);
	
	Mat img_erode = Morphology_erode(img_binary_crop, kernel);
	Mat img_erode_draw = img_draw_func(img_erode);
	
	//Mat img_tot; hconcat(img_binary_draw, img_dilate_draw, img_tot);
	Mat img_tot; hconcat(img_binary_draw, img_erode_draw, img_tot);
	
	namedWindow("asd", WINDOW_AUTOSIZE);
	imshow("img_binary_draw", img_binary_draw);
	imshow("img_dilate_draw", img_dilate_draw);
	waitKey();
	
	}

Mat Morphology_dilate(Mat& img, Mat& kernel) { 
	Mat img_dilate = Mat::zeros(img.size(), img.type());
	int kernelCenter = (kernel.rows - 1) / 2;
	
	Mat img_padded;  copyMakeBorder(img, img_padded, kernelCenter, kernelCenter, kernelCenter,  kernelCenter, BORDER_CONSTANT, 0);
	
	for (int i = kernelCenter; i < img_padded.rows - kernelCenter; i++) {
		for (int j = kernelCenter; j < img_padded.cols - kernelCenter; j++) {
			uchar maxVal = 0;

			for (int m = 0; m < kernel.rows; m++) {
				for (int n = 0; n < kernel.cols; n++) {
					uchar pixelVal = img_padded.at<uchar>(i + m - kernelCenter, j + n - kernelCenter);
					uchar kernelVal = kernel.at<uchar>(m, n);
					uchar result = pixelVal & kernelVal;

					if (result > maxVal) maxVal = result;
				}
			}
			img_dilate.at<uchar>(i-kernelCenter, j-kernelCenter) = maxVal;
		}
	}
	return img_dilate;
}

Mat Morphology_erode(Mat& img, Mat& kernel) { 
	Mat img_erode = Mat::zeros(img.size(), img.type());
	int kernelCenter = (kernel.rows - 1) / 2;

	Mat img_padded;  copyMakeBorder(img, img_padded, kernelCenter, kernelCenter, kernelCenter, kernelCenter, BORDER_CONSTANT, 255);

	for (int i = kernelCenter; i < img_padded.rows - kernelCenter; i++) {
		for (int j = kernelCenter; j < img_padded.cols - kernelCenter; j++) {
			uchar minVal = 255;

			for (int m = 0; m < kernel.rows; m++) {
				for (int n = 0; n < kernel.cols; n++) {
					uchar pixelVal = img_padded.at<uchar>(i + m - kernelCenter, j + n - kernelCenter);
					uchar kernelVal = kernel.at<uchar>(m, n);
					uchar result = pixelVal & kernelVal;

					if (result < minVal) minVal = result;
				}
			}
			img_erode.at<uchar>(i - kernelCenter, j - kernelCenter) = minVal;
		}
	}

	return img_erode;
}




Mat img_draw_func(Mat& img) { // 픽셀값 뽑아내기
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
