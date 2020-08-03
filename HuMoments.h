#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <math.h>

using namespace std;
using namespace cv;

class HuMomentsExtractor {

private:
	string outDir;
	int hMin, sMin, vMin;
	int hMax, sMax, vMax;
	int operationMode;

	Mat imageThreshold;

	// Hu Moments previously extracted for Red, Blue, and Green colors:

	double basehumoments[3][7] = { {0.272642,0.0336159,0.000377655,0.000337191,1.20246e-07,6.17523e-05,-4.39305e-09,},//ROJO
			{0.225426,0.0223209,0.000457201,4.43052e-05,5.09225e-09,5.13322e-06,-3.71903e-09},//AZUL
			{0.219967,0.0208571,4.66867e-05,3.98853e-06,2.42004e-11,2.4095e-07,-4.87509e-11} }; //VERDE

	// Red: hsv-min (0,53,162)    hsv-max (12,192,244)
	// Blue: hsv-min (83,125,183)   hsv-max (137,186,232)
	// Green: hsv-min (36,85,134) hsv-max (53,196,201)




	static void huFunc(int, void*);
	void printHSV();


	int euclideanDistance(vector<double>, int);

	void refreshImg();
	/*
	static void hMax(int,void*);
	static void sMax(int,void*);
	static void vMax(int,void*);
	static void hMin(int,void*);
	static void sMin(int,void*);
	static void vMin(int,void*);
	*/

public:
	HuMomentsExtractor(string = "fichero.txt");
	vector<double> extractHuMoments(Mat);
	vector<double> extractHuMoments(Mat, int, int, int, int, int, int);
	void capture();
	void setOperationMode(int);
};