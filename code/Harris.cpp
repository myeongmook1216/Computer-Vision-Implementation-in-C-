#include <iostream>
#include <opencv2/opencv.hpp>
#define PI 3.141592653589793238462643383279502884L

using namespace std;
using namespace cv;

Mat float_draw(Mat&);
Mat Gaussian_Kernel(int, double);
Mat conv_img(Mat&, Mat&);
Mat non_maximum_suppression(Mat&, Mat&);
int main() {
    //step 1 : 10x10 img 생성
    Mat img = (Mat_<double>(10, 10) <<
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    
    //step2
    Mat ux = (Mat_<double>(1, 3) << -1, 0, 1);
    Mat uy = ux.t();
    Mat dy = conv_img(img, uy);
    Mat dx = conv_img(img, ux);
    Mat dyy = dy.mul(dy);
    Mat dyx = dy.mul(dx);
    Mat dxx = dx.mul(dx);
    
    //step3 Gaussian filtering
    Mat g = Gaussian_Kernel(3, 1);
    Mat gdyy = conv_img(dyy, g);
    Mat gdxx = conv_img(dxx, g);
    Mat gdyx = conv_img(dyx, g);
    print(g);

    //step4 calculate C and non maximum suppression
    Mat C = (gdyy.mul(gdxx) - gdyx.mul(gdyx)) - 0.04 * ((gdxx + gdyy)).mul((gdxx + gdyy));
    Mat non_max = non_maximum_suppression(img, C);

    //draw img
    Mat img_draw = float_draw(img);
    Mat img_conv_draw_dx = float_draw(dx);
    Mat img_conv_draw_dy = float_draw(dy);
    Mat img_conv_draw_dyy = float_draw(dyy);
    Mat img_conv_draw_dyx = float_draw(dyx);
    Mat img_conv_draw_dxx = float_draw(dxx);
    Mat img_conv_draw_gdyy = float_draw(gdyy);
    Mat img_conv_draw_gdyx = float_draw(gdyx);
    Mat img_conv_draw_gdxx = float_draw(gdxx);
    Mat img_conv_draw_C = float_draw(C);
    Mat img_non_max_draw = float_draw(non_max);

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", img);

    namedWindow("Drawn Image", WINDOW_AUTOSIZE);
    imshow("Drawn Image", img_draw);
    imshow("dx", img_conv_draw_dx);
    imshow("dy", img_conv_draw_dy);
    imshow("dyy", img_conv_draw_dyy);
    imshow("dyx", img_conv_draw_dyx);
    imshow("dxx", img_conv_draw_dxx);
    imshow("gdyy", img_conv_draw_gdyy);
    imshow("gdyx", img_conv_draw_gdyx);
    imshow("gdxx", img_conv_draw_gdxx);
    imshow("C", img_conv_draw_C);
    imshow("non_max", img_non_max_draw);
    waitKey();
    return 0;
}

Mat non_maximum_suppression(Mat& img_origin, Mat& img_C) {
    Mat img_non_max = img_origin.clone();
    for (int y = 0; y < img_origin.rows; y++) {
        for (int x = 0; x < img_origin.cols; x++) {
            int count = 0;
            if (img_C.at<double>(y, x) > 0.1) {
                for(int m = -1; m < 2; m++){
                    for (int n = -1; n < 2; n++) {
                        if (img_C.at<double>(y, x) > img_C.at<double>(y + m, x + n)) count++;
                    }
                }
            }
            if (count == 8) img_non_max.at<double>(y, x) = 9;
        }
    }
    return img_non_max;
}


Mat conv_img(Mat& img, Mat& kernel) {
    int kernel_x = (kernel.cols - 1) / 2;
    int kernel_y = (kernel.rows - 1) / 2;
    Mat img_conv = img.clone();
    Mat img_padded; copyMakeBorder(img, img_padded, kernel_y, kernel_y, kernel_x, kernel_x, BORDER_CONSTANT, 0);
    for (int i = kernel_y; i < img_padded.rows - kernel_y; i++) {
        for (int j = kernel_x; j < img_padded.cols - kernel_x; j++) {
            double value = 0;
            for (int m = 0; m < kernel.rows; m++) {
                for (int n = 0; n < kernel.cols; n++) {
                    value += img_padded.at<double>(i - kernel_y + m, j - kernel_x + n) * (kernel.at<double>(m, n));
                }
            }

            img_conv.at<double>(i - kernel_y, j - kernel_x) = value;
        }
    }
    return img_conv;
}




Mat float_draw(Mat& img) {
    double scale1 = 512.0 / img.rows;   // 행의 scale
    double scale2 = 512.0 / img.cols;   // 열의 scale

    Mat img_draw(img.rows * scale1, img.cols * scale2, CV_32FC1, Scalar(0.0f));

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            string text = to_string(img.at<double>(i, j));
            int text_width, text_height;

            Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.3, 1, nullptr);
            text_width = text_size.width;
            text_height = text_size.height;
            int x = j * scale2 + (scale2 - text_width) / 2;
            int y = i * scale1 + (scale1 + text_height) / 2;

            putText(img_draw, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255), 1, LINE_AA);
        }
    return img_draw;
}



Mat Gaussian_Kernel(int size, double sigma) {
    Mat kernel(size, size, CV_64F);

    int kernelCenter = (size - 1) / 2;
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = kernelCenter - i;
            int y = kernelCenter - j;
            kernel.at<double>(i, j) = exp(-(static_cast<double>(x * x) + static_cast<double>(y * y)) / (2.0 * sigma * sigma)) / sqrt(2.0 * PI * sigma * sigma);
            sum += kernel.at<double>(i, j);
        }
    }
    kernel /= sum;
    return kernel;
}
