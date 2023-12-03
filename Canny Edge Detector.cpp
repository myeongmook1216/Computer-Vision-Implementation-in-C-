#include <iostream>
#include <opencv2/opencv.hpp>
#define PI 3.141592653589793238462643383279502884L

using namespace std;
using namespace cv;

Mat Gaussian_Kernel(int, float);
Mat img_pixel_save(Mat&, const string&);
Mat Sobel_operator(int); // 1넣으면 x operator반환
Mat conv(Mat&, Mat&);
Mat weight_x_y(Mat&, Mat&, float); // float의 크기는 x 가중치
Mat calc_gradient_magnitude_orientation(Mat&, Mat&, int); // 1넣으면 크기,다른숫자 넣으면 방향 반환
Mat NMS(Mat&, Mat&);
Mat Hysteresis_Threshold(Mat&, float, float);


int main() {
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
	if (img.empty()) { cout << "Image Open Error!" << endl;        return -1; } //img가 안열리면 -1 반환


	// Mat img_pixel = img_pixel_save(img, "mook_pixel.jpg");
	Mat gaussiankernel = Gaussian_Kernel(3, 1);
	Mat img_step1 = conv(img, gaussiankernel);

	Mat sobeloperator_x = Sobel_operator(1);	Mat sobeloperator_y = Sobel_operator(0);

	img_step1.convertTo(img_step1, CV_8UC1);
	Mat img_sobel_x = conv(img_step1, sobeloperator_x);	Mat img_sobel_y = conv(img_step1, sobeloperator_y);

	Mat img_sobel_x = conv(img_step1, sobeloperator_x);	Mat img_sobel_y = conv(img_step1, sobeloperator_y);

	Mat sobel_combined; hconcat(img_sobel_x, img_sobel_y, sobel_combined); 
	Mat sobel_weighted = weight_x_y(img_sobel_x, img_sobel_y, 0.5);
	

	Mat orientation = calc_gradient_magnitude_orientation(img_sobel_x, img_sobel_y, 0);
	Mat mag = calc_gradient_magnitude_orientation(img_sobel_x, img_sobel_y, 1);

	Mat img_NMS = NMS(mag, orientation);
	Mat img_Hyst = Hysteresis_Threshold(img_NMS, 50, 150);


	img_sobel_x.convertTo(img_sobel_x, CV_8UC1);
	img_sobel_y.convertTo(img_sobel_y, CV_8UC1);
	sobel_weighted.convertTo(sobel_weighted, CV_8UC1);
	mag.convertTo(mag, CV_8UC1);
	orientation.convertTo(orientation, CV_8UC1);
	img_NMS.convertTo(img_NMS, CV_8UC1);


	namedWindow("mook", WINDOW_AUTOSIZE);

	imshow("ot", orientation); 	imshow("mag", mag);
//	imshow("sobel_x", img_sobel_x); imshow("sobel_y" , img_sobel_y); 	imshow("sobel_weighted", sobel_weighted); 
	imshow("img_NMS", img_NMS); imshow("img_hyst", img_Hyst);
	waitKey();
	return 0;
}

Mat Hysteresis_Threshold(Mat& img_NMS, float lowThreshold, float highThreshold) {
	Mat img_H = Mat::zeros(img_NMS.rows, img_NMS.cols, CV_8U); int d = 1;
	Mat img_NMS_p; copyMakeBorder(img_NMS, img_NMS_p, d, d, d, d, BORDER_CONSTANT, 0);
	int flag = 0;

	for (int x = d; x < img_NMS_p.cols -d; ++x) {
		for (int y = d; y < img_NMS_p.rows - d ; ++y) {
			if (img_NMS_p.at<float>(y, x) > highThreshold) { img_H.at<uchar>(y - d, x - d) = 255;  }
			else if (img_NMS_p.at<float>(y, x) >= lowThreshold) {
				for (int m = -1; m <= 1; ++m) {
					for (int n = -1; n <= 1; ++n) {
						if (img_NMS_p.at<float>(y + m, x + n) >= highThreshold) {
							flag = true;
							break;
						}
					}
					if (flag) break;
				}
				img_H.at<uchar>(y - d, x - d) = (flag) ? 255 : 0;
			}

		else  {img_H.at<uchar>(y - d, x - d) = 0;}
		flag = 0;
		
		}
	}
	return img_H;
}


Mat NMS(Mat& mag, Mat& orient) {
	Mat img_NMS(mag.rows, mag.cols, CV_32F);
	Mat mag_p;
	int d = 1;
	int mode = 0;
	copyMakeBorder(mag, mag_p, d, d, d, d, BORDER_CONSTANT, 0);

	for (int x = d; x < mag_p.cols - d; ++x) {
		for (int y = d; y < mag_p.rows - d; ++y) {
			float value = orient.at<float>(y-d, x-d);

			if ((0 <= value && value < 22.5) || (157.5 <= value && value < 180))  mode = 1;
			else if (22.5 <= value && value < 67.5) mode = 2;
			else if (67.5 <= value && value < 112.5) mode = 3;
			else mode = 4;

			switch (mode) {
			case 1:
				if ((mag_p.at<float>(y, x) < mag_p.at<float>(y, x - 1)) || (mag_p.at<float>(y, x) < mag_p.at<float>(y, x + 1))) img_NMS.at<float>(y - d, x - d) = 0;
				else img_NMS.at<float>(y - d, x - d) = mag_p.at<float>(y, x);
				break;
			case 2:
				if ((mag_p.at<float>(y, x) < mag_p.at<float>(y + 1, x + 1)) || (mag_p.at<float>(y, x) < mag_p.at<float>(y - 1, x - 1))) img_NMS.at<float>(y - d, x - d) = 0;
				else img_NMS.at<float>(y - d, x - d) = mag_p.at<float>(y, x);
				break;
			case 3:
				if ((mag_p.at<float>(y, x) < mag_p.at<float>(y - 1, x)) || (mag_p.at<float>(y, x) < mag_p.at<float>(y + 1, x))) img_NMS.at<float>(y - d, x - d) = 0;
				else img_NMS.at<float>(y - d, x - d) = mag_p.at<float>(y, x);
				break;
			case 4:
				if ((mag_p.at<float>(y, x) < mag_p.at<float>(y - 1, x + 1)) || (mag_p.at<float>(y, x) < mag_p.at<float>(y + 1, x - 1))) img_NMS.at<float>(y - d, x - d) = 0;
				else img_NMS.at<float>(y - d, x - d) = mag_p.at<float>(y, x);
				break;
			}
		}
	}

	return img_NMS;
}



Mat calc_gradient_magnitude_orientation(Mat& img_x, Mat& img_y, int k) {
	Mat mag = Mat::zeros(img_x.rows, img_x.cols, CV_32F);
	Mat orient = Mat::zeros(img_x.rows, img_x.cols, CV_32F);

	for (int x = 0; x < img_x.cols; ++x) {
		for (int y = 0; y < img_y.rows; ++y) {
			mag.at<float>(y, x) = sqrt(img_x.at<float>(y, x) * img_x.at<float>(y, x) + img_y.at<float>(y, x) * img_y.at<float>(y, x));
			float a = atan2(img_y.at<float>(y, x), img_x.at<float>(y, x))*180 / PI;
			if (a < 0) a = -a; // -pi에서 pi로 받아온 각을 바로 윗줄에서 각으로 바꾼 뒤에, 각도의 절댓값 저장(0~180)
			orient.at<float>(y, x) =a;
		}
	}
	if (k == 1) return mag;
	else return orient;
}

Mat weight_x_y(Mat& img_x, Mat& img_y, float weight) {
	Mat img_w = Mat::zeros(img_x.rows, img_x.cols, CV_32F);
	for (int x = 0; x < img_x.cols; ++x) {
		for (int y = 0; y < img_y.rows; ++y) {
			img_w.at<float>(y,x) = img_x.at<float>(y, x) * weight + img_y.at<float>(y, x) * (1 - weight);
		}
	}
	return img_w;
}



Mat conv(Mat& img, Mat& kernel) {
	int kernelCenter = (kernel.rows - 1) / 2;
	Mat img_conv(img.cols - 2 * kernelCenter, img.rows - 2 * kernelCenter, CV_32F);

	for (int y = kernelCenter; y < img.rows - kernelCenter; y++) {
		for (int x = kernelCenter; x < img.cols - kernelCenter; x++) {
			float value = 0;
			for (int m = 0; m < kernel.rows; m++) {
				for (int n = 0; n < kernel.cols; n++) {
					value += static_cast<float>(img.at<uchar>(y - kernelCenter + m,
						x - kernelCenter + n)) * kernel.at<float>(m, n);
				}
			}
			if (value < 0) value = -value;

			img_conv.at<float>(y - kernelCenter, x - kernelCenter) =value;
		}
	}

	return img_conv;
}


Mat Gaussian_Kernel(int size, float sigma) {
	Mat kernel(size, size, CV_32F);

	int kernelCenter = (size - 1) / 2;
	float sum = 0.0;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int x = kernelCenter - i;
			int y = kernelCenter - j;
			kernel.at<float>(i, j) = exp(-(static_cast<float>(x * x) +
				static_cast<float>(y * y)) / (2.0 * sigma * sigma)) / sqrt(2.0 * PI * sigma * sigma);
			sum += kernel.at<float>(i, j);
		}
	}

	kernel /= sum;
	return kernel;
}

Mat Sobel_operator(int x) {
	Mat sobelKernelX = (Mat_<float>(3, 3) << -1, 0, 1,	-2, 0, 2,	-1, 0, 1);
	Mat sobelKernelY = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0,  1, 2, 1);
	if (x == 1) return sobelKernelX;
	else return sobelKernelY;
}



Mat img_pixel_save(Mat& img, const string& path) {
	const int scale = 20;
	Mat img_pixel = Mat::zeros(img.rows * scale, img.cols * scale, CV_8UC1);

	for (int x = 0; x < img.cols; ++x) {
		for (int y = 0; y < img.rows; ++y) {
			string text = to_string(static_cast<int>(img.at<uchar>(y, x)));
			Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.3, 1, nullptr);

			Point text_Pos(x * scale + scale / 4, y * scale + scale / 2);

			// 각 픽셀의 영역을 선으로 나눠줌
			rectangle(img_pixel, Rect(text_Pos.x, text_Pos.y - scale / 2, scale, scale), Scalar(255, 255, 255), 1);

			// 픽셀값의 글자 크기를 줄여줌
			putText(img_pixel, text, text_Pos, FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 1, LINE_AA);
		}
	}
	imwrite(path, img_pixel);

	return img_pixel;
}
