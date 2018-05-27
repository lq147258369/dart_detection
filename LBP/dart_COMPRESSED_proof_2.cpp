/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>

//std::string str = ss.str();    // unnecessary step, but shown to demonstrate how to obtain the string from stringstream.

using namespace std;
using namespace cv;

/** Function Headers */

float* detectAndDisplay(Mat frame);

float* Hough_Filter(Mat image,float* detection_info);

void Conv(
	cv::Mat &input,
	cv::Mat &ConvOutput, 
	cv::Mat &kernel);

void Normalise( cv:: Mat &input, 
		cv:: Mat &Normalised, 
		cv:: Mat &Reference);

float* GTvsDETECTION(Mat frame, float* detection_info, int* DART_info, int size);

void groundtruthGen(int* DART_dart_info, int number);//, Mat image2, Mat image3, Mat image4, Mat image5);

void groundtruthPrint(Mat image,int* DART_dart_info2, int number);//, Mat image2, Mat image3, Mat image4, Mat image5);

void PerformancePrint(float Correct_darts, float Detected_darts, int size,int i);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{	
	//GROUND TRUTH GENERATION
	float* CD_DD; //Correct darts & detected darts
	float* DD; //Detected darts
	float* DD_filtered;
	int Correct_darts = 0;
	int Detected_darts = 0;
	int Ground_Truth_darts = 0;
	//Strings generation
	std::stringstream ss;
	std::string str = ss.str();
	//Sizes array
	int array_sizes[16] = {3,3,3,3,3,3,3,3,6,3,9,3,3,3,6,3}; //Number of faces in the ground truth multiplied by 3
	for (int i=0;i<16;i++) {
		int *DART; 				//Create array to store the values of the ground truth picture
		DART = new int[array_sizes[i]]; 	//Matching size of the GT picture

		// 1. Read blank picture
		Mat frame = imread(argv[i+1], CV_LOAD_IMAGE_COLOR); 		
		//Mat frameHT = imread(argv[i+1], CV_LOAD_IMAGE_COLOR); 
		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Generate values for ground truth object in an array
		groundtruthGen(DART,i);

		// 4. Configure output names
		ss.str("");
		ss << "Detected" << i << ".jpg";
		str = ss.str();

		// 5. Detect darts and Display Result
		DD = detectAndDisplay(frame);
		//DD_filtered=Hough_Filter(frame,DD);
		CD_DD = GTvsDETECTION(frame,DD,DART,array_sizes[i]);
		PerformancePrint(CD_DD[0],CD_DD[1],array_sizes[i],i);

		// 5.5 Collecting data for final F1 Score
		Correct_darts = Correct_darts + CD_DD[0];
		Detected_darts = Detected_darts + CD_DD[1];
		Ground_Truth_darts+=(array_sizes[i]); //Not divided by 3 because the function already does

		// 6. Draw Ground Truth
		groundtruthPrint(frame,DART,i);

		// 7. Save and show final images
		imwrite( str, frame );
		namedWindow(str, CV_WINDOW_AUTOSIZE);
  		imshow(str, frame);
	  	waitKey(0);//wait for a key press until returning from the program
		delete[] DART;//Reset GT array
	  	frame.release();//free memory occupied by image 
	}
	//OVERALL F1 SCORE	
	PerformancePrint(Correct_darts,Detected_darts,Ground_Truth_darts,-1);
	return 0;
}

/////////////////////////////////////////////////////////////dart Detector//////////////////////////////////////////////////


float* detectAndDisplay( Mat frame)
{
	std::vector<Rect> darts;
	Mat frame_gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of darts found
	//std::cout << ""<< std::endl;
	//std::cout << "Detected darts: " << darts.size() << std::endl;

       // 4. DRAW BOX AROUND dartS

	/*for( int i = 0; i < darts.size(); i++ )
	{
		rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);

	}
	// 5. DARTS'CENTRE DRAWING

	for( int i = 0; i < darts.size(); i++ )
	{
		circle(frame, Point(darts[i].x + darts[i].width/2, darts[i].y + darts[i].height/2), 1, cvScalar(0, 255, 0), 2);
					
	}*/
	float * data =(float *)malloc(sizeof(float) * (2+darts.size()*3));
	data[0] = 0; //Reserved space for correct detected faces
	data[1] = (float)(darts.size());
	for (int d=0; d < darts.size();d++) {
		data[2+d*3]=darts[d].x;
		data[3+d*3]=darts[d].y;
		data[4+d*3]=(float)((darts[d].width+darts[d].height)/(float)(2));
	
	}
	return data;
}
/////////////////////////////////////////////////////////////Detection evaluator//////////////////////////////////////////////////
float* GTvsDETECTION(Mat frame, float* detection_info, int* DART_info, int size) {
	int t=0;
	if (frame.cols > frame.rows) {
	t = frame.rows/16; //+-7%
	}
	else {
	t = frame.cols/16;
	}
	//int t=50; //Threshold for ground truth PIXELS --> 677 pixels DART5 501 pixels DART15
	
	float Size_t = 0.20; // Size threshold PERCENTAGE +-20%

	int X_centre;
	int Y_centre;
	int Size;
	int Correct_darts=0;
       // 4. DRAW BOX AROUND dartS

	for( int i = 0; i < detection_info[1]; i++ )
	{
		rectangle(frame, Point(detection_info[2+i*3], detection_info[3+i*3]), Point(detection_info[2+i*3] + (int)(detection_info[4+i*3]), detection_info[3+i*3] + (int)(detection_info[4+i*3])), Scalar( 0, 255, 0 ), 2);

	}
	// 5. DARTS'CENTRE DRAWING

	for( int i = 0; i < detection_info[1]; i++ )
	{
		circle(frame, Point(detection_info[2+i*3] + (int)(detection_info[4+i*3])/2, detection_info[3+i*3] + (int)(detection_info[4+i*3])/2), 1, cvScalar(0, 255, 0), 2);
					
	}
	for( int i = 0; i < detection_info[1]; i++ )
	{

		for (int j=0;j < size/3;j++) {
			X_centre = DART_info[j*3+1]+DART_info[j*3]/2;
			Y_centre = DART_info[j*3+2]+DART_info[j*3]/2;


			if ((X_centre-t < (detection_info[2+i*3] + detection_info[4+i*3]/2)) && ((detection_info[2+i*3] + detection_info[4+i*3]/2) < X_centre+t) && (Y_centre-t < (detection_info[3+i*3] + detection_info[4+i*3]/2)) && ((detection_info[3+i*3] + detection_info[4+i*3]/2) < Y_centre+t)) {

			Size = DART_info[j*3];

				if ( ((float)Size*((float)(1-Size_t)) < detection_info[4+i*3]) && (detection_info[4+i*3] < (float)Size*((float)(1+Size_t))) && ((float)Size*((float)(1-Size_t)) < detection_info[4+i*3]) && (detection_info[4+i*3]< (float)Size*((float)(1+Size_t))) ) {

					Correct_darts++;

				}	
			}
		}
					
	}
	//static float data[2+darts.size()*3]; //Output

	detection_info[0] = (float)(Correct_darts);
	return detection_info;

}
/////////////////////////////////////////////////////////////Hough Filter//////////////////////////////////////////////////
float* Hough_Filter(Mat image,float* detection_info) {

	Mat matx = (Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat maty = (Mat_<double>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
  	
// CONVERT COLOUR to BW
	Mat gray_image;
 	cvtColor( image, gray_image, CV_BGR2GRAY );
	
// CONVOLUTE GRADIENTS
	Mat gradx_image;
	Conv(gray_image,gradx_image, matx); //Convolution with x gradient kernel
	Mat grady_image;
	Conv(gray_image,grady_image, maty); //Convolution with y gradient kernel
// OBTAINING GRADIENT MODULE AND PHASE

// MODULUS VARIABLES
	Mat gradModulus;
	gradModulus.create(gray_image.size(), CV_64F);
	Mat Modulus_image;
	Modulus_image.create(gray_image.size(), gray_image.type());
//PHASE VARIABLES
	Mat gradPhase;
	gradPhase.create(gray_image.size(), CV_64F);
	Mat Phase_image;
	Phase_image.create(gray_image.size(), gray_image.type());
//COMBINED LOOP FOR MADULUS AND PHASE
	for ( int i = 0; i < gradModulus.rows; i++ )
		{	
			for( int j = 0; j < gradModulus.cols; j++ )
			{
			//MODULUS CALCULATION
			gradModulus.at<double>(i,j) = std::sqrt(gradx_image.at<double>(i,j)*gradx_image.at<double>(i,j)+grady_image.at<double>(i,j)*grady_image.at<double>(i,j));
			//double      pow( double base, double exp ); IF NECESSARY

			//PHASE CALCULATION
			gradPhase.at<double>(i,j) = std::atan2(grady_image.at<double>(i,j),gradx_image.at<double>(i,j));
			//double      atan2( double y, double x );
			}
		}
	Normalise(gradModulus,Modulus_image,gray_image);
	Normalise(gradPhase,Phase_image,gray_image);
	//NORMALISATION OF GRADIENT X & Y
	Mat Ngradx_image;
	Normalise (gradx_image,Ngradx_image,gray_image);
	Mat Ngrady_image;
	Normalise (grady_image,Ngrady_image,gray_image);
////////////////////////////////////////////////HOUGH TRANSFORM//////////////////////////////////////////////////////////////////

//THRESHOLDING
	int T1 = 50; //70 as default value
	for ( int i = 0; i < Modulus_image.rows; i++ )
		{	
			for( int j = 0; j < Modulus_image.cols; j++ )
			{
				//Threshold
				if (Modulus_image.at<uchar>(i,j)>T1) {
				Modulus_image.at<uchar>(i,j) = 255;
				}
				else {
				Modulus_image.at<uchar>(i,j) = 0;
				}
			}
		}
//HOUGH SPACE
	//CREATION OF 3D MATRIX
	vector<vector<vector<double> > > Hough3D;
 	// Set up sizes. (HEIGHT x WIDTH)
	int radiusMax = 80; //default:80
	int radiusMin = 20;// not 1 BETTER A REASONABLE CIRCLE'S RADIUS ON THE IMAGE //default:20
  	Hough3D.resize(Modulus_image.rows+radiusMax*2);
  	for (int i = 0; i < Modulus_image.rows+radiusMax*2; ++i) {
    		Hough3D[i].resize(Modulus_image.cols+radiusMax*2);

    		for (int j = 0; j < Modulus_image.cols+radiusMax*2; ++j)
      			Hough3D[i][j].resize(radiusMax-radiusMin+1);
  	}
	
	for (int x = 0; x < Modulus_image.rows+radiusMax*2; x++ )
	{ 
		for (int y = 0; y < Modulus_image.cols+radiusMax*2; y++ )
		{
			for (int r = 0; r < radiusMax-radiusMin+1; r++)
			{
			Hough3D[x][y][r] = 0;
			}
		}	
	}
	//FILLING 3D HOUGH SPACE MATRIX
	double xoIntP=0;
	double yoIntP=0;
	double xoIntN=0;
	double yoIntN=0;
	//detection_info used [0,detections,y1,x1,size1,y2,x2,size2...]
	
	for (int w=0; w < detection_info[1]; w++) {
		for (int x = detection_info[3+w*3]; x < detection_info[3+w*3] + detection_info[4+w*3]; x++ )
		{ 
			for (int y = detection_info[2+w*3]; y < detection_info[2+w*3]+detection_info[4+w*3]; y++ )
			{
				if (Modulus_image.at<uchar>(x,y)==255) 
				{
					for (int r = radiusMin; r < radiusMax+1; r++)
					{
						for (int theta =-5; theta < 5; theta++) 
						{					
							double xoP = x + r * (sin(gradPhase.at<double>(x,y)+double(theta*M_PI/180)));//renormalise)
							double yoP = y + r * (cos(gradPhase.at<double>(x,y)+double(theta*M_PI/180)));
							double xoFractpartP = std::modf(xoP,&xoIntP);
							double yoFractpartP = std::modf(yoP,&yoIntP);
							if (xoFractpartP>0.5)
							xoIntP+=1;
							if (yoFractpartP>0.5)
							yoIntP+=1;
							Hough3D[xoIntP+radiusMax][yoIntP+radiusMax][r-radiusMin] +=1; 

							double xoN = x - r * (sin(gradPhase.at<double>(x,y)+double(theta*M_PI/180)));//renormalise)
							double yoN = y - r * (cos(gradPhase.at<double>(x,y)+double(theta*M_PI/180)));
							double xoFractpartN = std::modf(xoN,&xoIntN);
							double yoFractpartN = std::modf(yoN,&yoIntN);
							if (xoFractpartN>0.5)
							xoIntN+=1;
							if (yoFractpartN>0.5)
							yoIntN+=1;
							Hough3D[xoIntN+radiusMax][yoIntN+radiusMax][r-radiusMin] +=1;

						}
					}
				}
			}	
		}
	}
//PLOTTING IN 2D + SECOND THRESHOLDING

	Mat Hough2D;
	Hough2D.create(gray_image.size(), CV_64F);
	//Second Thresholding
///////////////////////////////////////////////////////////////standby////////////////////////////////////////////////////
	double minValue = 10000.0;
	double maxValue = -1000.0;
	double percentage = 0.75; //default

/*for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{
				for (int r = radiusMin; r < radiusMax+1; r++)
	
				{	
					if (Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]> maxValue) {

						maxValue = Hough3D[x+radiusMax][y+radiusMax][r-radiusMin];				
					}
					if (Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]< minValue) {

						minValue = Hough3D[x+radiusMax][y+radiusMax][r-radiusMin];				
					}


				}	
			}	
		}
for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{
				for (int r = radiusMin; r < radiusMax+1; r++)
	
				{	
					if (Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]> maxValue*percentage) {
						//circle(image, Point(y,x), r, cvScalar(255,0,0), 1);
						for (int i=0;i<360;i++) {
				image.at<Vec3b>((int)(x + r*sin(i)),(int)(y + r*cos(i)))[1] = 255;
				image.at<Vec3b>((int)(x + r*sin(i)),(int)(y + r*cos(i)))[2] = 0;
				image.at<Vec3b>((int)(x + r*sin(i)),(int)(y + r*cos(i)))[3] = 0;
						}

						//cout << "Circle"  << Point(x ,y) << endl;

					}
					//if (Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]< maxValue*percentage) {

						//Hough3D[x+radiusMax][y+radiusMax][r-radiusMin] = maxValue*percentage;
					//}

				}	
			}	
		}*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//ADDITION OF RADIUS VALUES
	for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{
				for (int r = radiusMin; r < radiusMax+1; r++)
	
				{
					Hough2D.at<double>(x,y)=Hough2D.at<double>(x,y) + Hough3D[x+radiusMax][y+radiusMax][r-radiusMin];
					//cout << Hough2D.at<double>(x,y) << endl;
/*					if (Hough2D.at<double>(x,y) < 0.5 ) {
					Hough2D.at<double>(x,y) = 0; 
					
					}*/
				}	
			}	
		}
	/*Mat Hough2Dlog;
	Hough2Dlog.create(gray_image.size(), CV_64F);
	double minV = 0;
	double maxV = 0;
	minMaxLoc(Hough2D, &minV,&maxV,NULL,NULL);
	for (int x = 0; x < Hough2D.rows; x++ )
		{ 
			for (int y = 0; y < Hough2D.cols; y++ )
			{	
				
				if ((Hough2D.at<double>(x,y))+(maxV-minV+1) <= 1) {
					Hough2Dlog.at<double>(x,y) = 0;
				}
				else {Hough2Dlog.at<double>(x,y)= std::log(std::log((Hough2D.at<double>(x,y))+(maxV-minV+1)));}
			}	
		}*/
	//cout << "MIN " << minV << " MAX " << maxV << endl;
	Mat Hough2DNorm;
	Normalise (Hough2D,Hough2DNorm,gray_image);
	//int T2 = 150;
	int T2 = 175;	
	double number_of_coins = 0;
	//Hough2D.at<double>(x,y)= std::exp(Hough2D.at<double>(x,y));
	//	

	for (int x = 0; x < Hough2DNorm.rows; x++ )
		{ 
			for (int y = 0; y < Hough2DNorm.cols; y++ )
			{
				if (Hough2DNorm.at<uchar>(x,y)>T2) {
				image.at<Vec3b>(x,y)[1] = Hough2DNorm.at<uchar>(x,y);
				image.at<Vec3b>(x,y)[2] = Hough2DNorm.at<uchar>(x,y);
				image.at<Vec3b>(x,y)[3] = Hough2DNorm.at<uchar>(x,y);
				//cout << "Point"  << Point(x ,y) << endl;

				}
				//int pixelvalue = Hough2DNorm.at<uchar>(x,y);
				/*if (pixelvalue > T2) {
				number_of_coins = number_of_coins + 1;
				}*/

			}	
		}
	//int attempt[(int)(detection_info[1])]; 				//Create array to store the values of the ground truth picture
	float * attempt =(float *)malloc(sizeof(float) * (2+detection_info[1]*3));
	int survivors=0;
	for (int w=0; w < detection_info[1]; w++) {
		for (int x = detection_info[3+w*3]; x < detection_info[3+w*3] + detection_info[4+w*3]; x++ )
		{ 
			for (int y = detection_info[2+w*3]; y < detection_info[2+w*3]+detection_info[4+w*3]; y++ )
			{

				if (Hough2DNorm.at<uchar>(x,y)>T2) {
				attempt[2+w*3]=1;	
				attempt[3+w*3]=1;
				attempt[4+w*3]=1;

				}	
			}
		}
					
	}
	for (int w=0; w < detection_info[1]; w++) {
		if(attempt[2+w*3]==1){
			survivors++;
		}
	}
	float * final_dart_info =(float *)malloc(sizeof(float) * (survivors*3+2));
	//float final_dart_info[survivor*3+2];
	final_dart_info[0] = detection_info[0];
	final_dart_info[1] = survivors;
	int counter=0;
	for (int w=0; w < detection_info[1]; w++) {
		if(attempt[2+w*3]==1){
			final_dart_info[2+counter*3] = detection_info[2+w*3];
			final_dart_info[3+counter*3] = detection_info[3+w*3];
			final_dart_info[4+counter*3] = detection_info[4+w*3];
			counter++;
		}
	}
	//SECOND THRESHOLDING
	//int T2=190; //20 without adding the radius dimension
//imwrite("final.png", gray_image);
 
//construct a window for image display
  namedWindow("Gradient X", CV_WINDOW_AUTOSIZE);
  namedWindow("Gradient Y", CV_WINDOW_AUTOSIZE);
  namedWindow("Gradient Module", CV_WINDOW_AUTOSIZE);
  namedWindow("Gradient Phase", CV_WINDOW_AUTOSIZE);
  namedWindow("B&W Image", CV_WINDOW_AUTOSIZE);
  //visualise the loaded image in the window
  imshow("Gradient X", Ngradx_image);
  imshow("Gradient Y", Ngrady_image);
  imshow("Gradient Module", Modulus_image);
  imshow("Gradient Phase", Phase_image);
  imshow("B&W Image", image);
  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image 
  Ngradx_image.release();
  Ngrady_image.release();
  Modulus_image.release();
  Phase_image.release();
  image.release();
	return final_dart_info;

}
//////////////////////////////////////////////////////Convolution function///////////////////////////////////////////////
void Conv(cv::Mat &input, cv::Mat &ConvOutput,cv::Mat &kernel)
{
	// intialise the output using the input

	//NORMALISATION BY DEFAULT
	//Mat Norm;
	//Norm.create(input.size(), CV_64F);	
	ConvOutput.create(input.size(), CV_64F);

	// create the kernel in 2D 

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		BORDER_REPLICATE );

	// now we can do the convolution
	double minValue = 0.0;
	double maxValue = 0.0; 
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			ConvOutput.at<double>(i,j) = sum;
		}
	}
	//NORMALISATION BY DEFAULT
	/*minMaxLoc(Norm, &minValue,&maxValue,NULL,NULL);
	for ( int i = 0; i < ConvOutput.rows; i++ )
		{	
		for( int j = 0; j < ConvOutput.cols; j++ )
			{	
			ConvOutput.at<uchar>(i,j) = (uchar)((Norm.at<double>(i,j)-minValue)*255/(maxValue-minValue));
			}
		}
	*/
}
/////////////////////////////////////Normalisation function////////////////////////////////////////
void Normalise( cv:: Mat &input, cv:: Mat &Normalised, cv:: Mat &Reference) {
	double minValue = 0.0;
	double maxValue = 0.0;
	Mat Norm; 
	Normalised.create(input.size(), Reference.type());
	minMaxLoc(input, &minValue,&maxValue,NULL,NULL);
	for ( int i = 0; i < input.rows; i++ )
		{	
		for( int j = 0; j < input.cols; j++ )
			{
			//NORMALISATION	
			Normalised.at<uchar>(i,j) = (uchar)((input.at<double>(i,j)-minValue)*255/(maxValue-minValue));
			}
		}

}
////////////////////////////////////////////////////////////////////////////////////////Ground truth Generation/////////////////////////////////////////////////////////////////////////////////////////////
	
void groundtruthGen(int* DART_dart_info, int number)
{
//////////////////////////////////////image0 + image1 + image3 + image4 + image5 + image6 + image7 + image9 + image11 + image12 + image13 + image15/////////////////////////////////////////////////////
	if ((number < 8)||(number == 9)||(number == 11)||(number == 12)||(number == 13)||(number == 15)) {
	int dart1[2];
	int size_dart1;
		if (number == 0) {
			dart1[0]=425;
			dart1[1]=13;
			size_dart1=180;	

		}
		if (number == 1) {
			dart1[0]=197;
			dart1[1]=131;
			size_dart1=192;	

		}
		if (number == 2) {
			dart1[0]=101;
			dart1[1]=96;
			size_dart1=91;	

		}
		if (number == 3) {
			dart1[0]=321;
			dart1[1]=150;
			size_dart1=70;	

		}
		if (number == 4) {
			dart1[0]=184;
			dart1[1]=97;
			size_dart1=193;	

		}
		if (number == 5) {
			dart1[0]=425;
			dart1[1]=142;
			size_dart1=107;		

		}
		if (number == 6) {
			dart1[0]=211;
			dart1[1]=116;
			size_dart1=66;		

		}
		if (number == 7) {
			dart1[0]=252;
			dart1[1]=170;
			size_dart1=146;	

		}
		if (number == 9) {
			dart1[0]=203;
			dart1[1]=48;
			size_dart1=234;		

		}
		if (number == 11) {
			dart1[0]=174;
			dart1[1]=103;
			size_dart1=60;		

		}
		if (number == 12) {
			dart1[0]=116;
			dart1[1]=76;
			size_dart1=138;	

		}
		if (number == 13) {
			dart1[0]=269;
			dart1[1]=118;
			size_dart1=138;	

		}
		if (number == 15) {
			dart1[0]=150;
			dart1[1]=56;
			size_dart1=140;	

		}


	
			//DART_dart_info = 
			int DART0[3] = {size_dart1,dart1[0],dart1[1]};

			for (int i=0; i<(sizeof((DART0))/sizeof((DART0[0]))); i++) {
				DART_dart_info[i] = DART0[i];
			}
	}


/////////////////////////////////image8 + image 14 /////////////////////////////////////////////////////////////////
	if ((number == 8)||(number == 14)) {
	int dart1[2];
	int size_dart1;
	int dart2[2];
	int size_dart2;
		if (number == 8) {
			dart1[0]=53;
			dart1[1]=255;
			size_dart1=86;	
			dart2[0]=842;
			dart2[1]=216;
			size_dart2=120;	
		}
		if (number == 14) {
			dart1[0]=118;
			dart1[1]=100;
			size_dart1=129;	
			dart2[0]=983;
			dart2[1]=93;
			size_dart2=129;	
		}

			//DART_dart_info = 
			int DART1[6] = {size_dart1,dart1[0],dart1[1],size_dart2,dart2[0],dart2[1]};

			for (int i=0; i<(sizeof((DART1))/sizeof((DART1[0]))); i++) {
				DART_dart_info[i] = DART1[i];
			}
	}
/////////////////////////////////image10 /////////////////////////////////////////////////////////////////


	if (number == 10) {
		int dart1[2]={86,104};
		int size_dart1=110;
		int dart2[2]={567,127};	
		int size_dart2=84;
		int dart3[2]={900,149};	
		int size_dart3=65;	

	
		//DART_dart_info = 
		int DART15[9] = {size_dart1,dart1[0],dart1[1],size_dart2,dart2[0],dart2[1],size_dart3,dart3[0],dart3[1]};

		for (int i=0; i<(sizeof((DART15))/sizeof((DART15[0]))); i++) {
			DART_dart_info[i] = DART15[i];
		}

	}



}

////////////////////////////////////////////////////////////////////////////////////////Ground truth printing/////////////////////////////////////////////////////////////////////////////////////////////

void groundtruthPrint( Mat image,int* DART_dart_info2, int number)
{
//////////////////////////////////////image0 + image1 + image3 + image4 + image5 + image6 + image7 + image9 + image11 + image12 + image13 + image15/////////////////////////////////////////////////////
	if ((number < 8)||(number == 9)||(number == 11)||(number == 12)||(number == 13)||(number == 15)) {

		//Draw box around darts found
		for( int i = 0; i < 1; i++ )
		{
			rectangle(image,Point(DART_dart_info2[i*3+1],DART_dart_info2[i*3+2]),Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3],DART_dart_info2[i*3+2]+DART_dart_info2[i*3]), Scalar( 255, 0, 0 ), 2);

		}

		// Draw centre of rectangles
		for( int i = 0; i < 1; i++ )
		{
			circle(image, Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3]/2,DART_dart_info2[i*3+2]+DART_dart_info2[i*3]/2), 1, cvScalar(255, 0, 0), 2);
		}
	}
/////////////////////////////////image8 + image 14 /////////////////////////////////////////////////////////////////
	if ((number == 8)||(number == 14)) {

		//Draw box around darts found
		for( int i = 0; i < 2; i++ )
		{
			rectangle(image,Point(DART_dart_info2[i*3+1],DART_dart_info2[i*3+2]),Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3],DART_dart_info2[i*3+2]+DART_dart_info2[i*3]), Scalar( 255, 0, 0 ), 2);

		}

		// Draw centre of rectangles
		for( int i = 0; i < 2; i++ )
		{
			circle(image, Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3]/2,DART_dart_info2[i*3+2]+DART_dart_info2[i*3]/2), 1, cvScalar(255, 0, 0), 2);
		}
	}
/////////////////////////////////image10 /////////////////////////////////////////////////////////////////


	if (number == 10) {

		//Draw box around darts found
		for( int i = 0; i < 3; i++ )
		{
			rectangle(image,Point(DART_dart_info2[i*3+1],DART_dart_info2[i*3+2]),Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3],DART_dart_info2[i*3+2]+DART_dart_info2[i*3]), Scalar( 255, 0, 0 ), 2);

		}

		// Draw centre of rectangles
		for( int i = 0; i < 3; i++ )
		{
			circle(image, Point(DART_dart_info2[i*3+1]+DART_dart_info2[i*3]/2,DART_dart_info2[i*3+2]+DART_dart_info2[i*3]/2), 1, cvScalar(255, 0, 0), 2);
		}

	}



}
void PerformancePrint(float Correct_darts, float Detected_darts, int size, int i) {
	//Strings generation
	std::stringstream ss;
	std::string str = ss.str();
	ss.str("");
	if (i>-1) {	
		ss << "Detected" << i << ".jpg";
		str = ss.str();
		std::cout << str << std::endl;
		std::cout << ""<< std::endl;
	}
	if (i<0) {
		std::cout << "FINAL F1 SCORE" << std::endl;
		std::cout << "______________" << std::endl;
	}
	std::cout << "Detected darts: " << Detected_darts << std::endl;
	std::cout << "True Positive darts: "<<  Correct_darts << std::endl;
	std::cout << "Ground truth darts: "<<  size/3 << std::endl;
	if (size/3 < Correct_darts) {
	Correct_darts = size/3;
	}
	float TPR = (float)Correct_darts/((float)size/3);
	std::cout << "TPR: "<<  TPR << std::endl;
	float Precision = (float)Correct_darts/((float)Detected_darts);
	std::cout << "Precision: "<<  Precision << std::endl;
	float F1_Score = (double)(2*TPR*Precision)/(double)(Precision+TPR);
	std::cout << "F1 Score: "<<  F1_Score << std::endl;
	std::cout << std::endl;
}
