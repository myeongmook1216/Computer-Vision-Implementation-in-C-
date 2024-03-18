#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void onMouse(int, int, int, int, void*); Point initialPoint(-1, -1); // initialPoint는 초기위치 저장변수
void calculateAndShowHistogram(Mat&);
Mat img_return_10_10(const Mat&); // 이미지를 60,60으로 자르고 화면에 출력하면서 자른 값 저장 
void displayPixelValues(const Mat& img);

int main(int argc, char** argv) {
    Mat img_BGR = imread("lena.jpg", IMREAD_COLOR); // 이미지 읽어 MAT에 대입 + cv::Vec3b 타입으로 각 픽셀이 저장된다. Vec3b는 3개의 8-bit 부호 없는 정수로 이루어진 벡터 즉, unsigned char로 이루어짐.  각 요소 -> B,G,R

    if (img_BGR.empty()) { cout << "Image Open Error!" << endl;        return -1; } //img가 안열리면 -1 반환

    //  cout << "행 출력" <<  img.rows << " 열 출력 " << img.cols << " 차원 출력 " << img.dims << "  채널 수 출력   " << img.channels() << " 행과 열 출력 " << img.size();

    Mat channels[3];   split(img_BGR, channels);    // 각 채널을 분리

    // imshow("Red Channel", channels[2]);    imshow("Green Channel", channels[1]);   imshow("Blue Channel", channels[0]); // R,G,B 채널 표시

    Mat img_resized;    resize(img_BGR, img_resized, Size(10, 10)); // img 크기 10x10으로 줄임

    //cout << "this is mat for 10x10x3 \n " << resizedImg; // 출력되는 행렬은 B G R B G R ... 총 30열 ; x 10행 으로 출력된다. [ 1 2 3 4 5 6  ... ;  12 2 3 4 5 6 7 ... ; ...] 여기서 1 2 3 은 각각 B G R
    Mat img_gray;    cvtColor(img_BGR, img_gray, COLOR_BGR2GRAY);                     // img를 gray로 바꾸기
    //Mat img_gray = 0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2];   //img를 gray로 바꾸기_함수구현

    Rect regiontocrop(10, 10, 3, 3); //잘라낼 영역 설정(x,y, width, height)
    // Mat img_cropped = img_BGR(regiontocrop); //img에서 영역을 잘라냄
    // imwrite("cropped_lena.jpg", img_cropped); // 잘라낸 이미지 현재 로컬폴더 위치에 저장
    Mat img_gray_cropped = img_gray(regiontocrop);   imwrite("img_gray_cropped.jpg", img_gray_cropped); // img_gray를 잘라내어 저장
    //영상에 도형 그리고 글자쓰기
    Mat img_draw = img_BGR;
    rectangle(img_draw, Point(100, 100), Point(200, 300), Scalar(0, 0, 255), 2);  // Point는 2D평면에서의 점을 나타내는데 사용됨, Scalar는 4차원 벡터를 나타낼 때 사용(최대 4개의 실수 값을 가짐)
    putText(img_draw, "Hello i am mook", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2); // 마지막 2는 선의 두께
    cout << img_gray_cropped;
    //img 출력 후 대기
    namedWindow("Original", WINDOW_AUTOSIZE);
    // imshow("Image out", img_draw); // display


    Mat originalImage = imread("lena.jpg", IMREAD_GRAYSCALE);

       // 회전 변환 매트릭스 생성 (예: 시계 방향으로 30도 회전)
    Mat rotationMatrix = getRotationMatrix2D(Point2f(originalImage.cols / 2, originalImage.rows / 2), 30.0, 1.0);

    // 역방향 매핑을 위한 역변환 매트릭스 생성
    Mat inverseRotationMatrix;
    invertAffineTransform(rotationMatrix, inverseRotationMatrix);

    // 역방향 매핑 수행
    Mat destinationImage;
    warpAffine(originalImage, destinationImage, inverseRotationMatrix, originalImage.size());



    /*
     Mat  img_a = img_BGR;
    imshow("Image drag to rectangle", img_a);
    setMouseCallback("Image drag to rectangle", onMouse, &img_a);  //함수 윈도우에 등록
    */    //마우스 드래그로 직사각형 그리기

    //waitKey(); // key 대기
    //calculateAndShowHistogram(img_gray);

  //  return 0;


    /*  //  모폴로지 openning
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat erosion_result;     erode(img_gray, erosion_result, kernel);
        Mat dilation_result;    dilate(img_gray, dilation_result, kernel);
        vector<Mat> image_horizontal = { img_gray, erosion_result, dilation_result}; // vector를 활용하여 Mat 객체를 담는 동적배열을 나타냄
        // vector는템플릿을 사용. 어떤 종류의 데이터든 저장할 수 있는 동적배열. vector가 Mat type의 객체를 담는 특정 종류의 동적 배열
        //벡터는 요소가 삽입되거나 삭제될 때 자동으로 크기를 조정하는 기능을 갖춘 동적 배열,  해당 저장 공간은 컨테이너에 의해 자동으로 처리
        //벡터 요소는 반복자를 사용하여 액세스하고 탐색할 수 있도록 연속 저장소에 배치
        Mat combined_result; hconcat(image_horizontal, combined_result); // inputarrayofarrays를 vector<Mat>을 통해 Mat의 배열을 만듦
        imshow("Combined_result", combined_result);
        waitKey(0);
        */


}

Mat img_return_10_10(const Mat& img) {
    Rect regiontocrop(30, 20, 10, 10);
    Mat img_10_10 = img(regiontocrop).clone();
    Mat img_resize;  resize(img_10_10, img_resize, Size(500, 500));
    return img_resize;
}

void displayPixelValues(const Mat& img) {
    Mat result = img.clone();

    for (int j = 0; j < img.rows; j++) {
        for (int i = 0; i < img.cols; i++) {
            // 픽셀 값을 문자열로 변환
            string pixel_value_str = to_string(static_cast<int>(img.at<uchar>(j, i)));

            // 픽셀 값 표시
            putText(result, pixel_value_str, Point(j * 500 / img.cols, i * 500 / img.rows), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 1);
        }
    }
    imshow("Image with Pixel Values", result);
    waitKey(0);
}



void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) { initialPoint = Point(x, y); }
    else if (event == EVENT_LBUTTONUP) {
        Mat img_a = *(Mat*)userdata;
        rectangle(img_a, initialPoint, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("Image drag to rectangle", img_a);
        initialPoint = Point(-1, -1);
    }
}

/*
    cout <<"이미지:\n"  << img<< endl; // 512x512x3 행렬이지만, 이걸로 출력하면 1차원 벡터로 출력됨
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cout << "[";
            for (int c = 0; c < img.channels(); ++c) {
                cout << static_cast<int>(img.at<Vec3b>(i, j)[c]);  //static_cast<int> -> C++에서 형변환 수행하는 연산자. img.at<Vec3b>(i, j)[c]는 이 객체에서 'c'인덱스에 해당하는 객체 가져옴. 'c'는 채널 의미
                if (c < img.channels() - 1) {
                    cout << ", ";
                }
            }
            cout << "] ";
        }
        cout << endl;
    }
  */

  /*
  void cv::calcHist(const Mat* images, // Histogram을 계산할 이미지들에 대한 배열이다.
      int           nimages, // images 배열에 포함된 이미지의 개수이다.
      const int* channels, //Histogram을 계산할 채널 번호들의 배열이다.
      InputArray    mask, //Histogram을 계산할 영역을 지정할 수 있다.
      OutputArray   hist, //Histogram 계산결과를 저장한다.
      int           dims, //Histogram 계산결과를 저장한 hist의 차원을 가리킨다.
      const int* histSize, //각 차원의 bin개수, 즉 빈도수를 분류할 칸의 개수를 의미한다.
      const float** ranges, //각 차원의 분류 bin의 최소값 최대값을 의미한다.
      bool          uniform = true,
      bool          accumulate = false
  )
  */

void calculateAndShowHistogram(Mat& img) {
    if (img.empty()) { cout << "이미지를 불러올 수 없습니다." << endl;        return; }
    int histSize = 256; // 보통 각 픽셀의 크기는 256이므로
    int hist[256] = { 0 };

    for (int j = 0; j < img.cols; ++j) {
        for (int i = 0; i < img.rows; ++i) {
            int pixel_value = static_cast<int>(img.at<uchar>(j, i));
            hist[pixel_value]++;
        }
    }
    // 히스토그램 시각화
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

    // 정규화
    int max_hist_value = *max_element(hist, hist + histSize);
    for (int i = 0; i < histSize; i++) {
        hist[i] = cvRound((double)hist[i] / max_hist_value * histImage.rows);
    }

    // 히스토그램 그리기
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - hist[i - 1]),
            Point(bin_w * i, hist_h - hist[i]),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    // 윈도우에 이미지와 히스토그램 출력
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", img);

    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);

    waitKey(0);
}
