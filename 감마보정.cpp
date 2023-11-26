#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat gamma_correction(Mat&, double );

int main() {
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
	
    double g_v_1 = 1.5; double g_v_3 = 3.0;
    Mat img_corrected_1 = gamma_correction(img, g_v_1);
    Mat img_corrected_3 = gamma_correction(img, g_v_3);
	namedWindow("asd", WINDOW_AUTOSIZE);
	imshow("img_gray", img);
    imshow("img_gray_gamma_1.5", img_corrected_1);
    imshow("img_gray_gamma_3", img_corrected_3);
    waitKey();
}

Mat gamma_correction(Mat& img, double gamma_value) {
    Mat img_normalized(img.rows, img.cols, CV_64F);
    Mat result(img.rows, img.cols, CV_8UC1);

    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++) {
            img_normalized.at<double>(y, x) = static_cast<double>(img.at<uchar>(y, x)) / 255.0;
        }
    

    for (int y = 0; y < img.rows; y++) 
        for (int x = 0; x < img.cols; x++) {
            img_normalized.at<double>(y, x) = pow(img_normalized.at<double>(y, x), gamma_value);
        }
    

    for (int y = 0; y < img.rows; y++) 
        for (int x = 0; x < img.cols; x++) {
            img_normalized.at<double>(y, x) *= 255.0;
        }
    

    for (int y = 0; y < img.rows; y++) 
        for (int x = 0; x < img.cols; x++) {
            result.at<uchar>(y, x) = static_cast<uchar>(img_normalized.at<double>(y, x));
        }
    

    return result;
}
