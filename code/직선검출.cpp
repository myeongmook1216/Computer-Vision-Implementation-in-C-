#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat hough_space_line(Mat&);
void findLocalMaxima(const Mat& src, Mat& dst, int neighborhoodSize, int threshold);
Mat draw_detected(Mat& img, const Mat& hough_space, int threshold);

int main() {
    Mat img_big = imread("building.png", IMREAD_GRAYSCALE);
    if (img_big.empty()) {
        cout << "img open error" << endl;
        return -1;
    }
    Mat img;
    resize(img_big, img, Size(img_big.cols / 2, img_big.rows / 2));

    // Step 1: Canny Edge Detection
    Mat edges;
    Canny(img, edges, 50, 150);

    // Step 2: find Hough space
    Mat hough_space = hough_space_line(edges);

    // Step 3: find local Maxima
    Mat localMaxima;
    findLocalMaxima(hough_space, localMaxima, 10, 130);

    // Step 4: draw detected lines
    Mat img_draw = draw_detected(img, localMaxima, 100);

    resize(hough_space, img, Size(hough_space.cols / 2, hough_space.rows / 2));
    resize(localMaxima, img, Size(localMaxima.cols / 2, localMaxima.rows / 2));
    // Display the result
    namedWindow("Canny Edge Detection", WINDOW_AUTOSIZE);
    imshow("img_origin", img);
    imshow("img_draw", img_draw);
    imshow("img_parameterspace", hough_space);
    imshow("img_detected_lines", localMaxima);
    waitKey(0);

    return 0;
}

Mat hough_space_line(Mat& img) {
    double max_rho = sqrt(pow(img.cols, 2) + pow(img.rows, 2));
    int max_rho_idx = cvRound(max_rho);
    Mat parameter_space = Mat::zeros(2 * max_rho_idx + 1, 180, CV_8UC1);

    for (int x = 0; x < img.cols; ++x) {
        for (int y = 0; y < img.rows; ++y) {
            if (img.at<uchar>(y, x) != 0) {
                for (int theta = 0; theta < 180; ++theta) {
                    // 변환된 파라미터 공간에서 직선의 방정식을 계산
                    double rho = x * cos(theta * CV_PI / 180) + y * sin(theta * CV_PI / 180);

                    // 직선이 이미지 내에 있는지 확인하고, 있다면 해당 위치의 값을 +1
                    int rhoIdx = cvRound(rho) + max_rho_idx;
                    if (rhoIdx >= 0 && rhoIdx < parameter_space.rows && (parameter_space.at<uchar>(rhoIdx, theta) != 255)) {
                        parameter_space.at<uchar>(rhoIdx, theta)++;
                    }
                }
            }
        }
    }

    return parameter_space;
}

void findLocalMaxima(const Mat& src, Mat& dst, int neighborhoodSize, int threshold) {
    dst = Mat::zeros(src.size(), CV_8UC1);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            int value = src.at<uchar>(y, x);
            bool isMaxima = true;

            for (int i = -neighborhoodSize; i <= neighborhoodSize; ++i) {
                for (int j = -neighborhoodSize; j <= neighborhoodSize; ++j) {
                    int neighborRow = y + i;
                    int neighborCol = x + j;

                    if (neighborRow >= 0 && neighborRow < src.rows && neighborCol >= 0 && neighborCol < src.cols) {
                        if (src.at<uchar>(neighborRow, neighborCol) > value) {
                            isMaxima = false;
                            break;
                        }
                    }
                }
                if (!isMaxima) {
                    break;
                }
            }

            if (isMaxima && value >= threshold) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}

Mat draw_detected(Mat& img, const Mat& hough_space, int threshold) {
    Mat draw_img = img.clone();
    for (int rhoIdx = 0; rhoIdx < hough_space.rows; ++rhoIdx) {
        for (int theta = 0; theta < hough_space.cols; ++theta) {
            if (hough_space.at<uchar>(rhoIdx, theta) > threshold) {
                // 변환된 파라미터 공간에서 직선의 방정식으로 역변환
                double rho = rhoIdx - hough_space.rows / 2.0;
                double theta_rad = theta * CV_PI / 180.0;
                double a = cos(theta_rad);
                double b = sin(theta_rad);
                double x0 = rho * a;
                double y0 = rho * b;

                // 좌표 계산
                Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
                Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

                // 직선 그리기
                line(draw_img, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
            }
        }
    }
    return draw_img;
}
