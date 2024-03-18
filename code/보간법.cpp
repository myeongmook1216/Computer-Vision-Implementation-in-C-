#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat bilinearInterpolation(const Mat&, float, float);

int main() {

    Mat inputImage = imread("lena.jpg", IMREAD_COLOR);

    if (inputImage.empty()) {
        cout << "이미지를 읽을 수 없습니다." << endl;
        return -1;
    }

    Mat backward_Magnify = bilinearInterpolation(inputImage,1.5,1.5);

    imshow("Original Image", inputImage);
    imshow("Backward Magnified Image", backward_Magnify);
    waitKey(0);

    return 0;
}

Mat bilinearInterpolation(const Mat& input, float scaleX, float scaleY) {
    int inputWidth = input.cols;    int inputHeight = input.rows;    
    int outputWidth = static_cast<int>(inputWidth * scaleX);
    int outputHeight = static_cast<int>(inputHeight * scaleY);

    Mat output = Mat::zeros(outputHeight, outputWidth, input.type());

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sourceX = x / scaleX;
            float sourceY = y / scaleY;

            int x1 = static_cast<int>(sourceX);
            int y1 = static_cast<int>(sourceY);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            float weightX = sourceX - x1;
            float weightY = sourceY - y1;

            x1 = min(max(x1, 0), inputWidth - 1);
            x2 = min(max(x2, 0), inputWidth - 1);
            y1 = min(max(y1, 0), inputHeight - 1);
            y2 =min(max(y2, 0), inputHeight - 1);

            output.at<Vec3b>(y, x) =
                (1 - weightX) * (1 - weightY) * input.at<Vec3b>(y1, x1) +
                weightX * (1 - weightY) * input.at<Vec3b>(y1, x2) +
                (1 - weightX) * weightY * input.at<Vec3b>(y2, x1) +
                weightX * weightY * input.at<Vec3b>(y2, x2);
        }
    }
    return output;
}
