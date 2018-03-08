#include <chrono>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Parameters
int whiteSensitivity = 50;
int colorsBlurKernelSize = 3;
int colorsTreshold = 1;
int edgesBlurKernelSize = 5;
int cannyLowTreshold = 60;
int cannyRatio = 3;
int cannyKernelSize = 3;

// Frames used by the program
Mat frame;
Mat croppedFrame;
Mat colors;
Mat yellow;
Mat white;
Mat edges;
Mat lanes;

// Other constant used in the process
bool debug = true;
Vec3b black(0,0,0);
Scalar yellowLow(15,0,160);
Scalar yellowHigh(30,255,255);
Scalar whiteLow(0,0,255-whiteSensitivity);
Scalar whiteHigh(255,whiteSensitivity,255);
const char* mainWindowName = "Lane detection";
const char* debugOriginalFrame = "Debug - original frame";
const char* debugCroppedFrame = "Debug - cropped frame";
const char* debugColorsFrame = "Debug - colors frame";
const char* debugEdgesFrame = "Debug - edges frame";

// Varibles used for setting ROI
Rect cropRect(0,0,0,0);
Point P1(0,0);
Point P2(0,0);
bool clicked = false;
const char* setROIWindowName = "Set ROI using mouse";
const char* ROIWindowName = "Selected ROI, if satisfying press 's'";

// Funtion shecks if the ROI rectangle has passed the frame boundaries
void checkBoundaries()
{
    if (cropRect.width > frame.cols - cropRect.x)
        cropRect.width = frame.cols - cropRect.x;

    if (cropRect.height > frame.rows  - cropRect.y)
        cropRect.height = frame.rows  - cropRect.y;

    if (cropRect.x < 0)
        cropRect.x = 0;

    if (cropRect.y < 0)
        cropRect.y = 0;
}

// Function shows selected ROI
void showROI()
{
    checkBoundaries();
    
    if (cropRect.width > 0 && cropRect.height > 0)
    {
        croppedFrame = frame(cropRect);
        imshow(ROIWindowName, croppedFrame);

        Mat imageWithRIO = frame.clone();
        rectangle(imageWithRIO, cropRect, Scalar(0,0,255), 3, 8, 0);
        imshow(setROIWindowName, imageWithRIO);
    }
}

// Callback function for mouse events which sets the ROI
void onMouse(int event, int mouseX, int mouseY, int flags, void* params)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            clicked = true;
            P1.x = mouseX;
            P1.y = mouseY;
            P2.x = mouseX;
            P2.y = mouseY;
            break;
        case CV_EVENT_LBUTTONUP:
            P2.x = mouseX;
            P2.y = mouseY;

            if (P1.x > P2.x)
            {
                cropRect.x=P2.x;
                cropRect.width=P1.x-P2.x; 
            }
            else 
            {
                cropRect.x=P1.x;
                cropRect.width=P2.x-P1.x; 
            }
            if (P1.y > P2.y)
            {
                cropRect.y=P2.y;
                cropRect.height=P1.y-P2.y;
            }
            else
            {
                cropRect.y=P1.y;
                cropRect.height=P2.y-P1.y;
            }

            clicked = false;
            showROI();
            break;
        default:
            break;
    }
}

Mat& detectColors(Mat& original)
{
    Mat& filtered(original);
    cvtColor(original, filtered, CV_BGR2HSV);

    MatIterator_<Vec3b> it, end;
    for( it = filtered.begin<Vec3b>(), end = filtered.end<Vec3b>(); it != end; ++it)
    {
        // Detect yellow
        if ((*it)[0] > 15 && (*it)[0] < 30 && (*it)[2] > 160)
            continue;
        // Detect bright white
        if ((*it)[2] > 0.6*255 && (*it)[1] < (-0.2*(*it)[1]/100 + 0.3)*255)
            continue;
        // Detect dark white
        if ((*it)[1] > 10 && (*it)[2] > ((*it)[1]+20)/100.0*255 && (*it)[2] < ((*it)[1]+40)/100.0*255)
            continue;

        (*it) = black;
    }

    cvtColor(filtered, filtered, CV_HSV2BGR);

    return filtered;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Not enough input parameters. First input parameter should be the path to video." << endl;
        return 1;
    }

    VideoCapture video(argv[1]);

    if (!video.isOpened())
    {
        cout << "Video could not be loaded." << endl;
        return 2;
    }
    video >> frame;

    double fps = video.get(CV_CAP_PROP_FPS);
    int numOfFrames = video.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Video FPS is " << fps << " and number of frames are " << numOfFrames << endl;

    namedWindow(setROIWindowName, WINDOW_NORMAL);
    resizeWindow(setROIWindowName, 900, 900);
    imshow(setROIWindowName, frame);
    setMouseCallback(setROIWindowName, onMouse, NULL);

    char c;
    while (1)
    {
        c = waitKey();

        if (c == 's' && croppedFrame.data)
        {
            destroyWindow(setROIWindowName);
            destroyWindow(ROIWindowName);
            destroyWindow(mainWindowName);
            break;
        }
        else if (c == 'q')
        {
            return 0;
        }
    }

    auto start = chrono::steady_clock::now();
    if (debug) namedWindow(debugCroppedFrame);
    namedWindow(mainWindowName);
    Mat test;
    while (1)
    {
        video >> frame;
        if (frame.empty())
            break;

        if (debug) imshow(debugOriginalFrame, frame);
        croppedFrame = frame(cropRect);
        colors = croppedFrame.clone();
        edges = croppedFrame.clone();
        if (debug) imshow(debugCroppedFrame, croppedFrame);
        
        blur(colors, colors, Size(colorsBlurKernelSize, colorsBlurKernelSize));
        cvtColor(colors, colors, CV_BGR2HSV);
        inRange(colors, yellowLow, yellowHigh, yellow);
        inRange(colors, whiteLow, whiteHigh, white);
        bitwise_and(yellow, white, colors);
        if (debug) imshow(debugColorsFrame, colors);
        if (debug) imshow("Yellow", yellow);
        if (debug) imshow("White", white);
        dilate(colors, colors, Mat());

        blur(edges, edges, Size(edgesBlurKernelSize, edgesBlurKernelSize));
        cvtColor(edges, edges, CV_BGR2GRAY);
        Canny(edges, edges, cannyLowTreshold, cannyLowTreshold*cannyRatio, cannyKernelSize);
        if (debug) imshow(debugEdgesFrame, edges);
        dilate(edges, edges, Mat());

        bitwise_and(colors, edges, lanes);
        cvtColor(lanes, lanes, CV_GRAY2BGR);
        lanes.copyTo(croppedFrame);
        imshow(mainWindowName, frame);

        // Press q on keyboard to exit
        char c = (char) waitKey(1/fps*1000);
        if( c == 'q' )
            break;
    }

    auto end = chrono::steady_clock::now();
    double elapsed_secs = chrono::duration_cast<chrono::milliseconds>(end - start).count() * 1.0 / 1000.0;
    cout << "Image processing took " << elapsed_secs << "s while video lasted for " << numOfFrames/fps << endl;

    video.release();
    destroyAllWindows();

    return 0;
}