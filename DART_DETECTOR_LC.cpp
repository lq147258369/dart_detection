/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - DART_DETECTOR_LINES.cpp
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
#include <stdio.h>


//std::string str = ss.str();    // unnecessary step, but shown to demonstrate how to obtain the string from stringstream.

using namespace std;
using namespace cv;

/** Function Headers */

float* detectAndDisplay(Mat frame);

float* Hough_Filter(Mat image,float* detection_info,Mat Hough, Mat TGRM);

void Conv(
	cv::Mat &input,
	cv::Mat &ConvOutput, 
	cv::Mat &kernel);

void Normalise( cv:: Mat &input, 
		cv:: Mat &Normalised, 
		cv:: Mat &Reference);

void Intersect(int T, cv::Mat Hough2DLinesNorm, cv::Mat VotesNorm);

float* GTvsDETECTION(Mat frame, float* detection_info, int* DART_info, int size);

void groundtruthGen(int* DART_dart_info, int number);//, Mat image2, Mat image3, Mat image4, Mat image5);

void groundtruthPrint(Mat image,int* DART_dart_info2, int number);//, Mat image2, Mat image3, Mat image4, Mat image5);

void PerformancePrint(float Correct_darts, float Detected_darts, int size,int i);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////MAIN FUNCTION///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
		Mat HoughSpace = Mat(frame.size(), CV_8UC1, Scalar(0));
		Mat Thresholded_Gradient = Mat(frame.size(), CV_8UC1, Scalar(0));		
		//Mat frameHT = imread(argv[i+1], CV_LOAD_IMAGE_COLOR); 
		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Generate values for ground truth object in an array
		groundtruthGen(DART,i);

		// 4. Configure output names
		ss.str("");
		ss << "Detected" << i << ".jpg";
		str = ss.str();

		// 5. Detect darts and Display Results
		DD = detectAndDisplay(frame);
		DD_filtered=Hough_Filter(frame,DD,HoughSpace,Thresholded_Gradient);
		CD_DD = GTvsDETECTION(frame,DD_filtered,DART,array_sizes[i]);
		PerformancePrint(CD_DD[0],CD_DD[1],array_sizes[i],i);

		// 6 Collecting data for final F1 Score
		Correct_darts = Correct_darts + CD_DD[0];
		Detected_darts = Detected_darts + CD_DD[1];
		Ground_Truth_darts+=(array_sizes[i]); //Not divided by 3 because the function already does

		// 7. Draw Ground Truth in the final image
		groundtruthPrint(frame,DART,i);

		// 8. Save final images and reset data array
		imwrite( str, frame );
		ss.str("");
		ss << "Hough_Space" << i << ".jpg";
		str = ss.str();
		imwrite( str, HoughSpace );
		ss.str("");
		ss << "Gradient_Modulus" << i << ".jpg";
		str = ss.str();
		imwrite( str, Thresholded_Gradient );
		frame.release();
		HoughSpace.release();
		Thresholded_Gradient.release();
		delete[] DART;//Reset GT array

	}
	//OVERALL F1 SCORE	
	PerformancePrint(Correct_darts,Detected_darts,Ground_Truth_darts,-1);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////VIOLA&JONES DETECTOR FUNCTION////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


float* detectAndDisplay( Mat frame)
{
	std::vector<Rect> darts;
	Mat frame_gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// 3. Save detections in a dynamic array
	float * data =(float *)malloc(sizeof(float) * (2+darts.size()*3));
	data[0] = 0; //Reserved space for correct detected faces
	data[1] = (float)(darts.size());
	for (int d=0; d < darts.size();d++) {
		data[2+d*3]=darts[d].x; //x is y for our representation
		data[3+d*3]=darts[d].y;
		data[4+d*3]=(float)((darts[d].width+darts[d].height)/(float)(2));
	
	}
	return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////HOUGH TRANSFORM FUNCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float* Hough_Filter(Mat image,float* detection_info,Mat Hough, Mat TGRM) {

	Mat matx = (Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat maty = (Mat_<double>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	//Mat image;

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
	//NORMALISATION OF GRADIENT MODULUS AND PHASE
	Normalise(gradModulus,Modulus_image,gray_image);
	Normalise(gradPhase,Phase_image,gray_image);

	//NORMALISATION OF GRADIENT X & Y
	Mat Ngradx_image;
	Normalise (gradx_image,Ngradx_image,gray_image);
	Mat Ngrady_image;
	Normalise (grady_image,Ngrady_image,gray_image);

////////////////////////////////////////////////HOUGH TRANSFORM METHOD//////////////////////////////////////////////////////////////////

// FIRST THRESHOLDING
	int T1 = 70; //70 as default value
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

	Modulus_image.copyTo(TGRM);///////Image to be saved

//FILLING 2D HOUGH SPACE MATRIX
	// Set up sizes. (roMax and ThetaMax)
	int Ro_resolution = 1;
	double RoSpan = 2*(Modulus_image.rows + Modulus_image.cols);
	int Theta_resolution = 1;
	double ThetaSpan = 180*Theta_resolution;
	Mat Hough2DLines = Mat(RoSpan,ThetaSpan, CV_64F, double(0));
//////////////////////Hough image combined to V&J/////////////////////////////////////////
//CONFINING SCANNING AREA
        	Mat HoughImage(gray_image.rows,gray_image.cols, CV_8UC3, Scalar(0,0,0));

	for (int w=0; w < detection_info[1]; w++) {
		for (int x = detection_info[3+w*3]; x < detection_info[3+w*3] + detection_info[4+w*3]; x++ )
		{ 
			for (int y = detection_info[2+w*3]; y < detection_info[2+w*3]+detection_info[4+w*3]; y++ )
			{
			HoughImage.at<Vec3b>(x,y)[1] = gray_image.at<uchar>(x,y);
			HoughImage.at<Vec3b>(x,y)[2] = 1; //Area for detected faces
			}	
		}
	}
//////////////////////////////
//VOTING LINES//////////////////
////////////////////////////
       double roInt = 0;;
	for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{
				if ((Modulus_image.at<uchar>(x,y)==255)&&(HoughImage.at<Vec3b>(x,y)[2] == 1)) 
				{
					for (int Theta = 0; Theta < ThetaSpan; Theta++) 
						{
							int ro = x * cos((double)((double)(Theta)/Theta_resolution) * M_PI/180) + y * sin((double)((double)(Theta)/Theta_resolution) * M_PI/180);
							if (ro > 0) {
							ro = RoSpan/2 + ro-1;
							Hough2DLines.at<double>(ro,Theta)+=1;
							}
							if (ro == 0) {
							ro = RoSpan/2-1;
							Hough2DLines.at<double>(ro,Theta)+=1;
							}
							if (ro < 0) {
							ro = RoSpan/2 + ro-1;
							Hough2DLines.at<double>(ro,Theta)+=1;
							}
						

					        
						 }
				}
			}
		}
//HOUGH SPACE NORMALISATION	
	Mat Hough2DLinesNorm;
	Normalise(Hough2DLines,Hough2DLinesNorm,gray_image);

//INTERSECTION VOTING
	Mat Intersection_Votes;
	Intersection_Votes.create(gray_image.size(), gray_image.type());
	int TH = 150;
	
	Intersect(TH, Hough2DLinesNorm, Intersection_Votes);
	//Intersection_Votes.copyTo(Hough);///////Image to be saved
///////////////////////////////////////////////////////
/////////////////VOTING CIRCLES//////////////////////////
////////////////////////////////////////////////////
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
	//Create regions

	//FILLING 3D HOUGH SPACE MATRIX
	double xoIntP=0;
	double yoIntP=0;
	double xoIntN=0;
	double yoIntN=0;
	
	//VOTING	
	for (int x = 0; x < HoughImage.rows; x++ )
		{ 
			for (int y = 0; y < HoughImage.cols; y++ )
			{
				if ((Modulus_image.at<uchar>(x,y)==255)&&(HoughImage.at<Vec3b>(x,y)[2] == 1)) 
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
//PLOTTING IN 2D + SECOND THRESHOLDING

	Mat Hough2DCircles = Mat(gray_image.size(), CV_64F, double(0));
	//Second Thresholding

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//ADDITION OF RADIUS VALUES TO CREATE 2D HOUGH SPACE
	for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{
				for (int r = radiusMin; r < radiusMax+1; r++)
	
				{
					//Hough2DCircles.at<double>(x,y)=Hough2DCircles.at<double>(x,y) + Hough3D[x+radiusMax][y+radiusMax][r-radiusMin];
					if (Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]>1) {
					Hough2DCircles.at<double>(x,y)=Hough2DCircles.at<double>(x,y) + (double)(Hough3D[x+radiusMax][y+radiusMax][r-radiusMin]);
					}
					else {
					Hough2DCircles.at<double>(x,y)=Hough2DCircles.at<double>(x,y) + 0;
					}

				}	
			}	
		}

	Mat Hough2DCirclesNorm;
	Normalise (Hough2DCircles,Hough2DCirclesNorm,gray_image);

/////////////////////////COMBINING VOTES//////////////////////////////
	//Intersection_Votes
	//Hough2DCircles

	Mat Hough2DCombined = Mat(gray_image.size(), CV_64F, double(0));
		for (int x = 0; x < Modulus_image.rows; x++ )
		{ 
			for (int y = 0; y < Modulus_image.cols; y++ )
			{


					Hough2DCombined.at<double>(x,y)=pow(1.15,((double)(Intersection_Votes.at<uchar>(x,y))+(double)(Hough2DCirclesNorm.at<uchar>(x,y))));

	
			}	
		}
	Mat Hough2DNorm;
	Normalise (Hough2DCombined,Hough2DNorm,gray_image);
	Hough2DNorm.copyTo(Hough);///////Image to be saved
///////////MAXIMUM VOTES FINDER///////////////
	int T2 = 175;
	int Maximum = 0;
	int dart_centre[2];
	float percentage_window=0.15; //A quarter of the columns of the original image
	int window_size = (HoughImage.cols)*percentage_window; //Fixed window to check summits
	for (int x = 0; x < HoughImage.rows; x++ )
		{ 
			for (int y = 0; y < HoughImage.cols; y++ )
			{
				//if (HoughImage.at<Vec3b>(x,y)[2] == 1)  //If the Hough Space is inside the V&J detected area then look for maximum values
				//{
					if (Hough2DNorm.at<uchar>(x,y)>T2) {
					image.at<Vec3b>(x,y)[1] = Hough2DNorm.at<uchar>(x,y);//Draw the votes in colour image
					image.at<Vec3b>(x,y)[2] = Hough2DNorm.at<uchar>(x,y);
					image.at<Vec3b>(x,y)[3] = Hough2DNorm.at<uchar>(x,y);
					//cout << "Point"  << Point(x ,y) << endl;
						for (int i=x-window_size;i<x+window_size;i++) { //Look for a particular window of the votes above threshold
							for (int j=y-window_size;j<y+window_size;j++) {
								if (Maximum < Hough2DNorm.at<uchar>(i,j)) {
									Maximum = Hough2DNorm.at<uchar>(i,j);
									dart_centre[0]=i; //Store the summit
									dart_centre[1]=j;	
								}				
							}
						}
						if (Maximum > 0) {
							HoughImage.at<Vec3b>(dart_centre[0],dart_centre[1])[3] = 1; //Assigned 1 in the third channel of the 3-channel hough space
							
						}
					Maximum = 0; //Reset maximum value and position
					dart_centre[0]=0;
					dart_centre[1]=0;

				//	}

				}

			}	
		}
	///////Count the number of detected darts(INTERSECTED POINTS)////////////////
	int number_of_darts=0;
	for (int x = 0; x < HoughImage.rows; x++ )
		{ 
			for (int y = 0; y < HoughImage.cols; y++ )
			{
				if (HoughImage.at<Vec3b>(x,y)[3] == 1) //The summits where stored in the third channel of the image
				{
					number_of_darts++;
				}
			}	
		}
	///////Store the position of the detected darts////////////////
	float * darts_centres =(float *)malloc(sizeof(float) * (number_of_darts*2));
	number_of_darts=0;
	for (int x = 0; x < HoughImage.rows; x++ )
		{ 
			for (int y = 0; y < HoughImage.cols; y++ )
			{
				if (HoughImage.at<Vec3b>(x,y)[3] == 1) 
				{
				darts_centres[number_of_darts*2]=x;
				darts_centres[number_of_darts*2+1]=y;
				circle(image, Point(y,x), 1, cvScalar(0,0,255), 2);
				number_of_darts++;
				}
			}	
		}

///////DISCARD VIOLA&JONES DETECTIONS WHICH DO NOT MATCH WITH THE HOUGH DETECTION WITHIN A THRESHOLD////////////////
				
	float * attempt =(float *)malloc(sizeof(float) * (2+detection_info[1]*3));//Array to store 0s or 1s depending on if they are valid as detection
	int survivors=0; //Remaining V&J detection as valid
	int v; //valid error
	if (image.cols > image.rows) {
	v = image.rows/16; //+-3.3%
	}
	else {
	v = image.cols/16;
	}
	for (int d=0; d < number_of_darts; d++) { // for "d" number of darts check if the match with the V&J detections
		for (int w=0; w < detection_info[1]; w++) {
			if ((darts_centres[d*2]-v<(detection_info[3+w*3] + detection_info[4+w*3]/2))&&(darts_centres[d*2]+v>(detection_info[3+w*3] + detection_info[4+w*3]/2))&&(darts_centres[d*2+1]-v<(detection_info[2+w*3] + detection_info[4+w*3]/2))&&(darts_centres[d*2+1]+v>(detection_info[2+w*3] + detection_info[4+w*3]/2))) //check if the coordinates are within a window threshold
			{ 
					attempt[2+w*3]=1;	
					attempt[3+w*3]=1;
					attempt[4+w*3]=1;	
			}
					
		}
	}
//////////////STORE THE NUMBER OF SURVIVORS////////////
	for (int w=0; w < detection_info[1]; w++) {
		if(attempt[2+w*3]==1){
			survivors++;
		}
	}
///////////////FINAL OUTPUT ARRAY FOR THE NEXT STAGE/////////////////////
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
	matx.release(); 
	maty.release();
	gray_image.release();
	gradx_image.release(); 
	grady_image.release(); 
	Ngradx_image.release(); 
	Ngrady_image.release(); 
	Modulus_image.release(); 
	gradModulus.release(); 
	Phase_image.release(); 
	gradPhase.release(); 
	Hough2DLines.release();
	Hough2DLinesNorm.release();
	Hough2DCircles.release();
	Hough2DCirclesNorm.release();
	Hough2DCombined.release();
	Hough2DNorm.release();
	Intersection_Votes.release();
	HoughImage.release();
	delete attempt;
	delete darts_centres;
	return final_dart_info;
	//return detection_info;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////INTERSECT LINES FUNCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Intersect(int T, cv::Mat Hough2DLinesNorm, cv::Mat VotesNorm) {
double y0;
double Yxmax;
double x0;
double Xxmax;
int number_of_lines=0;
for (int ro= 0; ro < Hough2DLinesNorm.rows; ro++ ) //ro
		{ 
			for (int theta = 0; theta < Hough2DLinesNorm.cols; theta++ )//Theta
			{

				if (Hough2DLinesNorm.at<uchar>(ro,theta)>T) { // (ro,Theta)
					number_of_lines++;// to calculate the size of container to storage the lines

				}

			}	
		}
float * lines =(float *)malloc(sizeof(float) * (number_of_lines*4));//size of vector which contains the lines
number_of_lines = 0;
for (int ro= 0; ro < Hough2DLinesNorm.rows; ro++ ) //ro
		{ 
			for (int theta = 0; theta < Hough2DLinesNorm.cols; theta++ )//Theta
			{

				if (Hough2DLinesNorm.at<uchar>(ro,theta)>T) // (ro,Theta)
				{ 

				if ((abs((double)(theta* M_PI/180)-(double)(M_PI/2)) > (double)(M_PI/9))) {
					x0=0;
					Xxmax=VotesNorm.cols;
					y0 = (double)((ro-(double)(Hough2DLinesNorm.rows/2-1))/cos((double)(theta* M_PI/180)));
					Yxmax = (double)(((ro-(double)(Hough2DLinesNorm.rows/2-1)) - (double)(Xxmax*sin((double)(theta* M_PI/180))))/cos((double)(theta* M_PI/180)));

				}
				else {
					y0=0;
					Yxmax=VotesNorm.rows;
					x0 = (double)((ro-(double)(Hough2DLinesNorm.rows/2-1))/sin((double)(theta* M_PI/180)));
					Xxmax = (double)(((ro-(double)(Hough2DLinesNorm.rows/2-1)) - (double)(Yxmax*cos((double)(theta* M_PI/180))))/sin((double)(theta* M_PI/180)));

				}
				

				//line(image, Point((int)(x0),(int)(y0)), Point((int)Xxmax,(int)(Yxmax)), cvScalar(255,0,0), 1);	
				lines[number_of_lines*4]=int(x0); //x0
				lines[number_of_lines*4+1]=int(y0);
				lines[number_of_lines*4+2]=int(Xxmax);
				lines[number_of_lines*4+3]=int(Yxmax);
				number_of_lines ++;
				}

			}	
		}

	//STORING VOTES//////////////////////
	Mat Voting_pool = Mat(VotesNorm.size(), CV_64F, double(0));
	Mat lines_store(VotesNorm.rows,VotesNorm.cols, CV_8UC3, Scalar(0,0,0));
//(VotesNorm.rows,VotesNorm.cols, CV_8UC3, Scalar(0,0,0));
		

	for (int r = 0; r < number_of_lines; r++)
	{
		x0=lines[r*4]; //x0
		y0=lines[r*4+1];
		Xxmax=lines[r*4+2];
		Yxmax=lines[r*4+3];
		line(lines_store, Point(x0,y0), Point(Xxmax,(int)(Yxmax)), cvScalar(1,1,1), 1);

		for(int x = 5;x < Voting_pool.rows-5; x++ ) 
			{
				for(int y = 5;y < Voting_pool.cols-5; y++ )
					{	
					Voting_pool.at<double>(x,y)=Voting_pool.at<double>(x,y)+lines_store.at<Vec3b>(x,y)[3];//3 represent first element of image

					}
			}
		line(lines_store, Point(x0,(int)(y0)), Point(Xxmax,Yxmax), cvScalar(0,0,0), 1);
	}
//////////////////Refine votes/////////////////////////////////
	/*for(int x = 0;x < Voting_pool.rows; x++ ) 
		{
			for(int y = 0;y < Voting_pool.cols; y++ )
				{	
					Voting_pool.at<double>(x,y)=pow(1.15,(int)(Voting_pool.at<double>(x,y)));//3 represent first element of image

				}
		}*/
	Mat Voting_Pool_Norm;
	Normalise(Voting_pool,Voting_Pool_Norm,Hough2DLinesNorm);
	Voting_Pool_Norm.copyTo(VotesNorm);///////Image to be saved

	/*int T_inter=0;
		for (int x = 0; x < Voting_Pool_Norm.rows; x++ )
		{ 
			for (int y = 0; y < Voting_Pool_Norm.cols; y++ )
			{
				if (Voting_Pool_Norm.at<uchar>(x,y)>T_inter) {
				image.at<Vec3b>(x,y)[1] = Voting_Pool_Norm.at<uchar>(x,y);
				image.at<Vec3b>(x,y)[2] = Voting_Pool_Norm.at<uchar>(x,y);
				image.at<Vec3b>(x,y)[3] = Voting_Pool_Norm.at<uchar>(x,y);


				}


			}	
		}*/
	Voting_pool.release();
	lines_store.release();
	delete lines;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////DETECTIONS EVALUATOR//////.//////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float* GTvsDETECTION(Mat frame, float* detection_info, int* DART_info, int size) {
	int t=0;
	if (frame.cols > frame.rows) {
	t = frame.rows/16; //+-7%
	}
	else {
	t = frame.cols/16;
	}
	
	float Size_t = 0.20; // Size threshold PERCENTAGE +-20%

	int X_centre;
	int Y_centre;
	int Size;
	int Correct_darts=0;
       //  DRAW BOX AROUND DARTS

	for( int i = 0; i < detection_info[1]; i++ )
	{
		rectangle(frame, Point(detection_info[2+i*3], detection_info[3+i*3]), Point(detection_info[2+i*3] + (int)(detection_info[4+i*3]), detection_info[3+i*3] + (int)(detection_info[4+i*3])), Scalar( 0, 255, 0 ), 2);

	}
	//  DARTS'CENTRE DRAWING

	for( int i = 0; i < detection_info[1]; i++ )
	{
		circle(frame, Point(detection_info[2+i*3] + (int)(detection_info[4+i*3])/2, detection_info[3+i*3] + (int)(detection_info[4+i*3])/2), 1, cvScalar(0, 255, 0), 2);
					
	}
	int control=0;
	for (int j=0;j < size/3;j++) {
	
		for( int i = 0; i < detection_info[1]; i++ )
		{


			X_centre = DART_info[j*3+1]+DART_info[j*3]/2;
			Y_centre = DART_info[j*3+2]+DART_info[j*3]/2;


			if ((X_centre-t < (detection_info[2+i*3] + detection_info[4+i*3]/2)) && ((detection_info[2+i*3] + detection_info[4+i*3]/2) < X_centre+t) && (Y_centre-t < (detection_info[3+i*3] + detection_info[4+i*3]/2)) && ((detection_info[3+i*3] + detection_info[4+i*3]/2) < Y_centre+t)) {

			Size = DART_info[j*3];

				if ( ((float)Size*((float)(1-Size_t)) < detection_info[4+i*3]) && (detection_info[4+i*3] < (float)Size*((float)(1+Size_t))) && ((float)Size*((float)(1-Size_t)) < detection_info[4+i*3]) && (detection_info[4+i*3]< (float)Size*((float)(1+Size_t))) ) {

					if (control==0) {//This is to avoid repeated validation for the same ground truth checking
					Correct_darts++;
					control=1;
					}

				}	
			}
		
		}
	control=0;
					
	}

	detection_info[0] = (float)(Correct_darts);
	return detection_info;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////CONVOLUTION FUNCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////NORMALISATION FUNCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////GROUND TRUTH ARRAY GENERATOR FUNCTION////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////GROUND TRUTH DRAWER FUNCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////F1SCORE CALCULATOR FUNCTION//////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
