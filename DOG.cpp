#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#define PI 3.141592653589793238462643383279502884L

using namespace std;
using namespace cv;

vector<vector<Mat>> create_Multiple_Scale_Images(Mat&); 
Mat conv_img(Mat& img, Mat& kernel);
Mat Gaussian_Kernel(int size, double sigma);

int main() {
    Mat img_origin = imread("RV.jpg", IMREAD_GRAYSCALE);
    Mat img_resized;     resize(img_origin, img_resized, Size(), 0.13, 0.13, INTER_NEAREST);
    img_resized.convertTo(img_resized, CV_64F);
    Mat GK = Gaussian_Kernel(3, 4.8253);
    Mat img_conv = conv_img(img_resized, GK);
   vector<vector<Mat>> Multiple_Scale_images = create_Multiple_Scale_Images(img_resized);
       
    img_resized.convertTo(img_resized, CV_8UC1);
    img_conv.convertTo(img_conv, CV_8UC1);

   Multiple_Scale_images[0][3].convertTo(Multiple_Scale_images[0][3], CV_8UC1);

   Mat blurredImage;
   GaussianBlur(img_resized, blurredImage, Size(3,3), 4.8253, 4.8253);

   


   //DOG 구하는 과정
    Mat img_1 = Multiple_Scale_images[0][0] - Multiple_Scale_images[0][1];
  img_1.convertTo(img_1, CV_8UC1, 255);
   //Mat img_2 = Multiple_Scale_images[0][1] - Multiple_Scale_images[0][2];
   //img_2.convertTo(img_2, CV_8UC1, 255);
   //Mat img_3 = Multiple_Scale_images[0][2] - Multiple_Scale_images[0][3];
   //img_3.convertTo(img_3, CV_8UC1, 255);

   //Mat img_4 = Multiple_Scale_images[0][3] - Multiple_Scale_images[0][4];
   //img_4.convertTo(img_4, CV_8UC1, 255);
   
   for (int i = 0; i < 4; i++) {
       for (int j = 0; j < 6; j++) {
           Multiple_Scale_images[i][j].convertTo(Multiple_Scale_images[i][j], CV_8UC1);
       }
   }



   Mat img_l = Multiple_Scale_images[0][2] - Multiple_Scale_images[0][3];
   img_l.convertTo(img_l, CV_8UC1, 255);


   Mat concat_img_0;    Mat concat_img_1;    Mat concat_img_2;    Mat concat_img_3; 
   hconcat(Multiple_Scale_images[0], concat_img_0);
   hconcat(Multiple_Scale_images[1], concat_img_1);
   hconcat(Multiple_Scale_images[2], concat_img_2);
   hconcat(Multiple_Scale_images[3], concat_img_3);
    namedWindow("Drawn Image", WINDOW_AUTOSIZE);

    imshow("DOG_1", img_1); // imshow("DOG_2", img_2);
   // imshow("DOG_5", img_4); 
    imshow("DOG_3", img_l);
    
            //imshow("DOG_4", img_4);
    /*
    imshow("octave_1", concat_img_0);
    imshow("octave_2", concat_img_1);
    imshow("octave_3", concat_img_2);
    imshow("octave_4", concat_img_3);
    */
    imshow("bluuredImage", blurredImage);
    waitKey();
    return 0;
}

vector<vector<Mat>> create_Multiple_Scale_Images(Mat& img_origin) {
    Mat img = img_origin.clone();
    vector<vector<Mat>> octaveVector;
    double k = pow(2.0, 1.0 / 3.0);
    
    
    for (int octave = 0; octave < 4; octave++) {
        vector<Mat> scale_space_Vector;
        double next_sigma = 1.5199;
        
        for (int s = 0; s < 6; s++) {
            double current_sigma = next_sigma;
            Mat Gaussian_kernel =Gaussian_Kernel(3, current_sigma);

            cout << "next_sigma :  " << current_sigma << " " << endl;
           scale_space_Vector.push_back(conv_img(img, Gaussian_kernel));
           next_sigma = k * current_sigma;
        }
         
        octaveVector.push_back(scale_space_Vector);
        resize(octaveVector[octave][3], img, Size(), 0.5, 0.5, INTER_NEAREST);

    }

    return octaveVector;

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
