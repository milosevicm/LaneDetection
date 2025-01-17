#include <chrono>
#include <ctime>
#include <iostream>
#include <numeric>
#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Parameters
double recombineThreshold = 20;
double minSlopeDetection = tan(20*CV_PI/180);
double maxSlopeDetection = tan(60*CV_PI/180);
int whiteSensitivity = 120;
int colorsBlurKernelSize = 3;
int colorsTreshold = 1;
int edgesBlurKernelSize = 3;
int cannyLowTreshold = 40;
int cannyRatio = 3;
int cannyKernelSize = 3;
int houghTreshold = 50;
int houghLength = 35;
int houghGap = 5;
Scalar yellowLow(15,100,100);
Scalar yellowHigh(40,255,255);
Scalar whiteLow(0,0,255-whiteSensitivity);
Scalar whiteHigh(255,whiteSensitivity,255);
Mat dilateElement = getStructuringElement(MORPH_RECT, Size(3,5));

// Shared variables
vector<Vec4i>   houghLanes;
vector<double>  leftSlopes;
vector<int>     leftXIntercepts;
vector<Vec4i>   leftLines;
vector<double>  rightSlopes;
vector<int>     rightXIntercepts;
vector<Vec4i>   rightLines;
vector<double>  noiseSlopes;
vector<int>     noiseIntercepts;
double          leftSlope;
int             leftXIntercept;
bool			leftFound;
double          rightSlope;
int             rightXIntercept;
bool			rightFound;
double          lastAverageSlope;
int             lastAverageIntercept;
double          lastSlope;
int             lastIntercept;

// Frames used by the program
Mat frame;
Mat croppedFrame;
Mat colors;
Mat yellow;
Mat white;
Mat edges;
Mat lanes;

// Other constant used in the process
bool debug = false;
const char* mainWindowName = "Lane detection";

// Varibles used for setting ROI
int halfOfROIWidth = 0;
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

// Method that displays lines on the main frame
void displayLines(vector<double> slopes, vector<int> intercepts, Scalar color)
{
    assert(slopes.size() == intercepts.size());

    for (int i = 0; i < slopes.size(); i++)
    {
        line(frame, Point((cropRect.height+slopes[i]*intercepts[i])/slopes[i] +cropRect.x, cropRect.y),
                Point(intercepts[i]+cropRect.x, cropRect.height+cropRect.y), color, 3, 8 );
    }

}

// Method that separates noise lines, if no line is separated returns false
bool recombine(vector<double>& slopes, vector<int>& intercepts, double found)
{
    assert(slopes.size() == intercepts.size());

    lastAverageSlope = accumulate(slopes.begin(), slopes.end(), 0.0)/slopes.size();
    lastAverageIntercept = accumulate(intercepts.begin(), intercepts.end(), 0.0)/intercepts.size();

    double maxDist = 0;
    int maxIdx = 0;
    double distance = 0;

    for (size_t i = 0; i < slopes.size(); i++)
    {
        // Criteria function, punishing lines that are too offset
        distance = abs(slopes[i] - ((1+3*!found)*lastAverageSlope+3*found*lastSlope)/4)*200
            + abs(intercepts[i]-((1+!found)*lastAverageIntercept+found*lastIntercept)/2);

        if (distance > maxDist)
        {
            maxIdx = i;
            maxDist = distance;
        }
    }

    if (maxDist > recombineThreshold)
    {
        noiseSlopes.push_back(slopes[maxIdx]);
        noiseIntercepts.push_back(intercepts[maxIdx]);
        slopes.erase(slopes.begin() + maxIdx);
        intercepts.erase(intercepts.begin() + maxIdx);
        return true;
    }

    return false;
}

// Method that clasifies lines obtained by Hough transformation
void clasify(vector<Vec4i> lines)
{
    leftFound = leftSlopes.size() != 0;
    leftSlopes.clear();
    leftXIntercepts.clear();
    leftLines.clear();
    rightFound = rightSlopes.size() != 0;
    rightSlopes.clear();
    rightXIntercepts.clear();
    rightLines.clear();
    noiseSlopes.clear();
    noiseIntercepts.clear();

    // Devide set into left and right lanes
    for( size_t i = 0; i < lines.size(); i++ )
    {
        // Slope calculated in "human-seen" coordinating system
        double k = (double)(lines[i][1]-lines[i][3]) / (double)(lines[i][2]-lines[i][0]);

        // Reject lines that are not in range of 20..60 degrees
        if (abs(k) > minSlopeDetection && abs(k) < maxSlopeDetection)
        {
            // Value of x for y = 0 in "human-seen" coordinating system
            int intercept = lines[i][0]-(cropRect.height-lines[i][1])/k;

            if (intercept < halfOfROIWidth)
            {
                leftSlopes.push_back(k);
                leftXIntercepts.push_back(intercept);
                leftLines.push_back(lines[i]);
            }
            else
            {
                rightSlopes.push_back(k);
                rightXIntercepts.push_back(intercept);
                rightLines.push_back(lines[i]);
            }
        }    
    }

    // Reject once that are offseting to much
    if (leftSlopes.size() > 0)
    {
    	lastSlope = leftSlope;
    	lastIntercept = leftXIntercept;
        while (recombine(leftSlopes, leftXIntercepts, leftFound));
        leftSlope = lastAverageSlope;
        leftXIntercept = lastAverageIntercept;
        if (debug) displayLines(leftSlopes, leftXIntercepts, Scalar(0, 255, 255));
        line(frame, Point((cropRect.height+leftSlope*leftXIntercept)/leftSlope +cropRect.x, cropRect.y),
                Point(leftXIntercept+cropRect.x, cropRect.height+cropRect.y), Scalar(255,0,0), 3, 8 ); 
    }

    // Reject once that are offseting to much
    if (rightSlopes.size() > 0)
    {
    	lastSlope = rightSlope;
    	lastIntercept = rightXIntercept;
        while (recombine(rightSlopes, rightXIntercepts, rightFound));
        rightFound = true;
        rightSlope = lastAverageSlope;
        rightXIntercept = lastAverageIntercept;
        if (debug) displayLines(rightSlopes, rightXIntercepts, Scalar(0, 255, 255));
        line(frame, Point((cropRect.height+rightSlope*rightXIntercept)/rightSlope +cropRect.x, cropRect.y),
                Point(rightXIntercept+cropRect.x, cropRect.height+cropRect.y), Scalar(0,255,0), 3, 8 );  
    }

    if (debug) displayLines(noiseSlopes, noiseIntercepts, Scalar(0, 0, 255));
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Not enough input parameters. First input parameter should be the path to video." << endl;
        return 1;
    }

    debug = argc > 2;

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
            halfOfROIWidth = cropRect.width/2;
            assert(halfOfROIWidth > 0);
            break;
        }
        else if (c == 'q')
        {
            return 0;
        }
    }

    auto start = chrono::steady_clock::now();
    namedWindow(mainWindowName);

    while (1)
    {
        video >> frame;
        if (frame.empty())
            break;

        croppedFrame = frame(cropRect);
        colors = croppedFrame.clone();
        edges = croppedFrame.clone();
        
        // Colors processing
        blur(colors, colors, Size(colorsBlurKernelSize, colorsBlurKernelSize));
        cvtColor(colors, colors, CV_BGR2HSV);
        inRange(colors, yellowLow, yellowHigh, yellow);
        inRange(colors, whiteLow, whiteHigh, white);
        bitwise_or(yellow, white, colors);
        if (debug) imshow("Debug - Yellow", yellow);
        if (debug) imshow("Debug - White", white);
        dilate(colors, colors, dilateElement);
        if (debug) imshow("Debug - Colors", colors);

        // Edges processing
        blur(edges, edges, Size(edgesBlurKernelSize, edgesBlurKernelSize));
        cvtColor(edges, edges, CV_BGR2GRAY);
        Canny(edges, edges, cannyLowTreshold, cannyLowTreshold*cannyRatio, cannyKernelSize);
        if (debug) imshow("Debug - Edges", edges);
        dilate(edges, edges, dilateElement);

        bitwise_and(colors, edges, lanes);
        if (debug) imshow("Debug - Lanes", lanes);
        HoughLinesP(lanes, houghLanes, 1, CV_PI/180, houghTreshold, houghLength, houghGap);

        // Separate lines into left, right and noise
        clasify(houghLanes);

        // Show the result
        imshow(mainWindowName, frame);

        // Press q on keyboard to exit
        char c;
        if (debug)
            c = (char) waitKey();
        else
            c = (char) waitKey(1/fps*1000);

        if( c == 'q' )
            break;
    }

    // Performance check
    auto end = chrono::steady_clock::now();
    double elapsed_secs = chrono::duration_cast<chrono::milliseconds>(end - start).count() * 1.0 / 1000.0;
    cout << "Image processing took " << elapsed_secs << "s while video lasted for " << numOfFrames/fps << endl;

    video.release();
    destroyAllWindows();

    return 0;
}