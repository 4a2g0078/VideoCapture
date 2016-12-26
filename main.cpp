#pragma warning(disable:4819)
#pragma warning(disable:4996)
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
//#include <math.h>
#include <Windows.h>
#include <time.h>

using namespace cv;
using namespace std;

Mat bgframe;
Mat accumulator;
Mat facedetection;
Mat fgn2;
String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
vector<Rect> faces_tmp;

int counter = 0;


void Background_Prepare()
{
	Mat frame;
	string filename = "out.avi";
	//int count = 1;
	float alpha = 0.1;
	VideoCapture backgroundImg(filename);
	if (!backgroundImg.isOpened())
	{
		cout << "Can't open video" << endl;
		return;
	}

	bool bFirstBG = true;
	while (backgroundImg.read(frame))
	{
		if (frame.empty())
		{
			break;
		}
		if (bFirstBG)
			frame.copyTo(bgframe);
		else {
			bgframe = bgframe * (1 - alpha) + frame * alpha;
		}
	}
}

void facedetectDisplay(Mat image)
{
	for (size_t i = 0; i < faces_tmp.size(); i++)
	{
		Point center(faces_tmp[i].x + faces_tmp[i].width / 2, faces_tmp[i].y + faces_tmp[i].height / 2);
		ellipse(image, center, Size(faces_tmp[i].width / 2, faces_tmp[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);


		//Mat faceROI = frame_gray(faces[i]);
		//std::vector<Rect> eyes;

		////-- In each face, detect eyes
		//eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		//for (size_t j = 0; j < eyes.size(); j++)
		//{
		//	Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
		//	int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		//	circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		//}
	}
	//-- Show what you got
	//imshow(window_name, frame);
}

size_t detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	faces_tmp = faces;
	
	return faces.size();
}

void Background_compare(Mat frame)
{	
	Mat defBG,BlurImg,diff ,dst, kernel;

	//defBG = imread("default.jpg");

	size_t currentBG_faces, currentFrame_faces;
	currentBG_faces = detectAndDisplay(bgframe);
	currentFrame_faces = detectAndDisplay(frame);
	//(frame, mat1, CV_BGR2GRAY);
	//cvtColor(bgframe, mat2, CV_BGR2GRAY);
	//absdiff(mat1, mat2, dst);
	//absdiff(defBG, bgframe, diff);
	//cvtColor(diff, diff, CV_BGR2GRAY);
	//threshold(diff, dst, 20, 255, THRESH_BINARY);
	////bitwise_xor(frame, bgframe, dst);
	//int n1 = countNonZero(dst);
	GaussianBlur(defBG, BlurImg, Size(3, 3), 0, 0);
	absdiff(BlurImg, bgframe, diff);
	cvtColor(diff, diff, CV_BGR2GRAY);
	threshold(diff, dst, 20, 255, THRESH_BINARY);
	kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(dst, dst, MORPH_CLOSE, kernel);
	morphologyEx(dst, dst, MORPH_OPEN, kernel);
	fgn2 = dst * 0.5;

	int n2 = countNonZero(dst);

	/*absdiff(frame, defBG, diff);
	cvtColor(diff, diff, CV_BGR2GRAY);
	threshold(diff, dst, 20, 255, THRESH_BINARY);

	int n3 = countNonZero(dst);*/

	if (currentBG_faces > 0)
	{
		if (currentFrame_faces > 0)
		{
			/*if (n1 > n2 && n2 < n3)
				bgframe = defBG;				
			else if(n1 < n2)
				return;*/
		}
		else
		{
			//bgframe = defBG;
		}
		
	}
	else {
		if (currentFrame_faces > 0)
		{
			/*if (n2 > n3)
				return;
			else if (n2 < n3 )
				bgframe = defBG;*/
		}
		else
		{
			/*if (n1 > 0)
			{
				if (n2 > n3)
					bgframe = defBG;
				else if (n2 < n3)
					return;
			}*/
		}
	}
	/*if (n1 < n2)
	{
		bgframe = defBG;
	}
	else if (n1 )
		imwrite("default.jpg", bgframe);*/
}
//void Background_Subtraction(int num)
//{
//	stringstream ss;
//	Mat *img = new Mat[num];
//	for (int i = 0; i < num; i++)
//	{
//		ss << "capture_" << i + 1 << ".jpg";
//		ss >> filename;
//		img[i] = imread(filename);
//		ss.clear();
//		if (img[i].empty())
//		{
//			cout << "Unable to read image" << endl;
//			return;
//		}
//	}
//	/*img1 = imread("capture_1.jpg");
//	img2 = imread("capture_2.jpg");
//	img3 = imread("capture_3.jpg");*/
//	//img4 = imread("capture_4.jpg");
//	/*if (img1.empty() || img2.empty() || img3.empty())
//	{
//	cout << "Unable to read image" << endl;
//	return;
//	}*/
//
//	//while (true)
//	//{
//		for (int i = 0; i < num; i++)
//		{
//			ProcessImages(bgframe, img[i], i+1);
//		}
//		/*ProcessImages(bgframe, img1, count++);
//		ProcessImages(bgframe, img2, count++);
//		ProcessImages(bgframe, img3, count++);*/
//		//ProcessImages(bgframe, img4, count++);
//
//		//accumulateWeighted(frame, bgframe, 0.003, 0);
//		//count = 1;
//		/*GaussianBlur(img1, img1, Size(3, 3), 0, 0);
//		absdiff(frame, img1, diffimg);
//		imshow("img1", diffimg);*/
//		
//		//if (waitKey(10) == 27)
//		//	break;
//	//}
//	delete[] img;
//	
//
//	return;
//}
void Contour_Filling(Mat& mask, Mat& dst)
{
	int niters = 3;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src = mask.clone();

	//bitwise_not(mask, src);
	//dilate(mask, src, Mat(), Point(-1, -1), niters);
	//erode(src, src, Mat(), Point(-1, -1), niters * 2);
	//dilate(src, src, Mat(), Point(-1, -1), niters);

	//findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	findContours(src, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	dst = Mat::zeros(mask.size(), CV_8UC1);

	if (contours.size() == 0)
		return;

	int idx = 0, largestComp = 0;
	double maxArea = 0;

	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		if (area > maxArea)
		{
			maxArea = area;
			largestComp = idx;
		}
		
	}
	int idex = 0;

	drawContours(src, contours, largestComp, Scalar(255), FILLED, LINE_8, hierarchy);

	src.copyTo(dst);
	//bitwise_not(src, dst);
}

void fillEdgeImage(Mat& mask, Mat& dst)
{
	Mat mask_INV = mask.clone();
	floodFill(mask_INV, Point(0, 0), Scalar(255));
	bitwise_not(mask_INV, mask_INV);
	dst = (mask_INV | mask);
}

Mat getForegroundMask(Mat image, Mat background, double th)
{
	Mat fgMask, kernel, fgMask_th, BlurImg, BurBg, Ycbcr_Img, Ycbcr_Bg,diffImg, diffImg2, fgMask_fill;
	Mat img_channel[3],bg_channel[3];
	//double dist;
	float alpha = 0.01;
	int iterations = 3;

	GaussianBlur(image, BlurImg, Size(3, 3), 0, 0);
	GaussianBlur(background, BurBg, Size(3, 3), 0, 0);
	//absdiff(BlurImg, BurBg, diffImg);
	cvtColor(BlurImg, Ycbcr_Img, CV_BGR2YCrCb);
	cvtColor(BurBg, Ycbcr_Bg, CV_BGR2YCrCb);

	split(Ycbcr_Img, img_channel);
	split(Ycbcr_Bg, bg_channel);
	//absdiff(image, background, diffImg);
	//absdiff(Ycbcr_Img, Ycbcr_Bg, diffImg);
	absdiff(img_channel[1], bg_channel[1], diffImg);
	absdiff(img_channel[2], bg_channel[2], diffImg2);
	//diffImg = diffImg * 0.5 + diffImg2 * 0.5;
	diffImg = diffImg + diffImg2 ;
	fgMask_th = Mat::zeros(diffImg.size(), CV_8UC1);
	//addWeighted(BurBg, alpha, diffImg, 1 - alpha, 0, bgframe);
	/*split(diffImg, img_channel);
	diffImg = img_channel[1] + img_channel[2];*/
	
	//cvtColor(diffImg, diffImg, CV_BGR2GRAY);
	
	threshold(diffImg, fgMask_th, th, 255, THRESH_BINARY);
	
	kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	////erode(fgMask_th, fgMask_th, kernel, cvPoint(-1, -1), 3);
	//morphologyEx(fgMask_th, fgMask_th, MORPH_CLOSE, kernel, cvPoint(-1, -1), 1);
	//morphologyEx(fgMask_th, fgMask, MORPH_OPEN, kernel, cvPoint(-1, -1), 1);

	dilate(fgMask_th, fgMask_th, kernel, Point(-1, -1), iterations);
	erode(fgMask_th, fgMask_th, kernel, Point(-1, -1), iterations * 2);
	dilate(fgMask_th, fgMask, kernel, Point(-1, -1), iterations);
	
	//Contour_Filling(fgMask_th, fgMask);
	//Contour_Filling(fgMask_th, fgMask_fill);
	//fillEdgeImage(fgMask_fill, fgMask);

	//fgMask = fgMask_th.clone();
	//cvtColor(BurBg, BurBg, CV_YCrCb2BGR);
	//cvtColor(BlurImg, BlurImg, CV_BGR2GRAY);
	//cvtColor(BurBg, BurBg, CV_BGR2GRAY);
	
	//threshold(fgMask, fgMask_th, 220, 255, THRESH_BINARY_INV);
	/*try
	{
		for (int j = 0; j<diffImg.rows; ++j)
			for (int i = 0; i<diffImg.cols; ++i)
			{
				cv::Vec3b pix = diffImg.at<cv::Vec3b>(j, i);

				dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
				dist = sqrt(dist);

				if (dist>th)
				{
					fgMask_th.at<unsigned char>(j, i) = 255;
				}
			}
	}
	catch (exception ex)
	{

	}*/
	
	return fgMask;

}

void ProcessImages(Mat background, Mat image, int& framecount)
{
	static int count = 0;
	float alpha = 0.00001;
	//Mat updateBG;
	Mat fgMask;
	Mat fgImg;
	Mat scaled;

	stringstream ss;
	string str;
	
	//size_t facesize;
	
	image.copyTo(fgImg);	
	fgMask = getForegroundMask(fgImg, background, 10);
	fgImg.create(image.size(), image.type());
	fgImg = Scalar::all(0);
 	image.copyTo(fgImg, fgMask);
	image.copyTo(facedetection);

	//accumulateWeighted(image, background, alpha);
	//addWeighted(background, alpha, image, 1 - alpha, 0, background);

	//accumulateWeighted(image, accumulator, 0.003);
	//convertScaleAbs(accumulator, scaled);

	//if (framecount > 120)
	//{
	//	convertScaleAbs(accumulator, bgframe);
	//	//bgframe = bgframe * alpha + image * (1 - alpha);
	//	//framecount = 0;
	//}else
	//	framecount++;

	/*Mat ImgHSV, skinImg_HSV;
	Mat ImgYCrCb, skinImg_YCrCb, mask;
	cvtColor(image, ImgHSV, CV_BGR2HSV);
	cvtColor(image, ImgYCrCb, CV_BGR2YCrCb);
	inRange(ImgHSV, cv::Scalar(0, 58, 20), cv::Scalar(50, 174, 230), skinImg_HSV);
	inRange(ImgYCrCb, Scalar(50, 133, 77), Scalar(255, 173, 127), skinImg_YCrCb);*/

	detectAndDisplay(image);
	//facedetectDisplay(fgImg);
	//facedetectDisplay(fgMask);
	//facedetectDisplay(facedetection);

	//ss << "Image now_" << count++;
	ss << "Image";
	ss >> str;
	if (count == 0) {
		namedWindow(str);
		moveWindow(str, fgImg.cols, 0);
	}
	
	imshow(str, fgImg);
	ss.clear();
	//ss << "Mask now_" << count;
	ss << "Mask";
	ss >> str;
	if (count == 0) {
		namedWindow(str);
		moveWindow(str, fgImg.cols*2, 0);
		
		//namedWindow("face_detection");
		//moveWindow("face_detection", 0, fgImg.rows);
		/*namedWindow("skin_HSV");
		moveWindow("skin_HSV", 0, fgImg.rows);
		namedWindow("skin_YCrCb");
		moveWindow("skin_YCrCb", fgImg.cols, fgImg.rows);*/
	}

	//capture fg image
	/*if (framecount == 0)
	{

		int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
		double fontScale = 2;
		int thickness = 2;
		Point textOrg(fgMask.size().width / 2, 30);
		ss.clear();
		ss << counter;
		string someText;
		ss >> someText;
		putText(fgMask, someText, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
		imshow(str, fgMask);
		ss.clear();
		ss << "capture_mask\\mask_" << counter++ << ".jpg";
		ss >> str;
		imwrite(str, fgMask);
		framecount++;
	}
	else if (framecount < 5)
	{
		framecount++;
	}
	else
		framecount = 0;*/
	int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
	double fontScale = 2;
	int thickness = 2;
	Point textOrg(fgMask.size().width / 2, 30);
	ss.clear();
	ss << framecount++;
	string someText;
	ss >> someText;
	putText(fgMask, someText, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
	imshow(str, fgMask);

	//imshow("face_detection", facedetection);
	/*if (count == 0) {
		namedWindow("Weighted Average");
		moveWindow("Weighted Average", 0, fgImg.rows);
	}
	imshow("Weighted Average", scaled);*/
	/*imshow("skin_HSV", skinImg_HSV);
	imshow("skin_YCrCb", skinImg_YCrCb);*/
}

int main()
{
	string streamaddress = "rtsp://root:pass@192.168.0.51/axis-media/media.amp";
	//string streamaddress = "http://root:pass@192.168.0.51/axis-cgi/mjpg/video.cgi";
	//VideoCapture cap(0);
	VideoCapture cap(streamaddress);

	if (!cap.isOpened()) {
		cout << "Unable to open the camera\n";
		return -1;
	}
	namedWindow("Camera");
	//namedWindow("MOG");
	//waitKey(2000);

	double FPS = 30;
	cout << "FPS:" << FPS << endl;
	//int delay = 1000.0 / FPS;
	//clock_t start ;
	//bool isDefaultRecord = false;
	//bool isRecord = false;
	//bool isCaptureImage = false;
	//int capture_num = 3
	int count = 1;
	
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.0); // 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480.0); // 720);
	
	// Get the width/height of the camera frames
	int frame_width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	VideoWriter video;
	VideoWriter defaultBackground;
	Mat frame;
	accumulator = Mat::zeros(frame.size(), CV_32FC3);
	video.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), FPS, Size(frame_width, frame_height), true);
	bool flagErr = false;
	int bgFrameCount = 0;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			flagErr = true;
			cout << "Can't read frames from your camera" << endl;
			break;
		}
		imshow("Camera", frame);

		char c = cvWaitKey(10);
		if (c == 13)
		{
			cout << "capture" << endl;
			imwrite("default.jpg", frame);
		}
			
		if (c != 32 && bgFrameCount==0)
			continue;
		if ( bgFrameCount==0)
			cout << "Start Record" << endl;
		
		
		video.write(frame);

		
		bgFrameCount++;
		if (bgFrameCount > 60)
			break;
	}
	video.release();
	cout << "Stop record" << endl;
	Beep(523, 500);
	if (flagErr) {
		return -1;
	}
	
	Background_Prepare();
	//Load the cascade
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); };

	cap >> frame;
	if (frame.empty()) {
		flagErr = true;
		cout << "Can't read frames from your camera" << endl;
		return -1;
	}
	//Background_compare(frame);
	waitKey(1000);
	cvMoveWindow("Camera", 0, 0);
	int framecount = 0;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			flagErr = true;
			cout << "Can't read frames from your camera" << endl;
			break;
		}
		
		//video.write(frame);
		imshow("Camera", frame);
		//imshow("MOG", fgMaskMOG);
		
		int key = waitKey(5);
		if (key == 27) // Stop the camera if the user presses the "ESC" key		
			break;
		if (counter > 20)
			break;
		//Stat record background before beep sound if the user presses "Space" key
		//if (key == 32)
		//{
		//	cout << "Start Record" << endl;
		//	video.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), FPS, Size(frame_width, frame_height), true);
		//	isRecord = true;	
		//	start = clock();
		//}
		//Record default background if the user presses "Enter" key
		//if (key == 13)
		//{
		//	destroyAllWindows();
		//	cout << "Start DefaultRecord" << endl;
		//	defaultBackground.open("defaultBackground.avi", CV_FOURCC('M', 'J', 'P', 'G'), FPS, Size(frame_width, frame_height), true);
		//	isDefaultRecord = true;
		//	start = clock();
		//}
		//if (isDefaultRecord)
		//{
		//	cout << clock() - start << endl;
		//	defaultBackground.write(frame);
		//	if (clock() - start >= 4000)
		//	{
		//		cout << "Stop record" << endl;
		//		isDefaultRecord = false;
		//		defaultBackground.release();
		//	}
		//}
		//if (isRecord)
		//{
		//	cout << clock() - start << endl;
		//	//waitKey(delay);
		//	video.write(frame);
		//	if (clock() - start >= 3000)
		//	{
		//		cout << "Stop record" << endl;
		//		Beep(523, 500);
		//		waitKey(1500);
		//		isRecord = false;
		//		isCaptureImage = true;
		//		cout << "Start capture image" << endl;
		//		start = clock();
		//		video.release();
		//		//break;
		//	}
		//}
		
 	//	if (isCaptureImage)
		//{

			string filename;
			stringstream str(filename);
			str << "capture\\capture_" << count << ".jpg";
			str >> filename;
		

			//if (clock() - start >= 2000* count)
			//{
			//	cout << filename << endl;
				imwrite(filename, frame);
			//	Beep(659,200);
				count++;
			//}				
			//if (count > capture_num)
			//{
			//	count = 1;
			//	isCaptureImage = false;
			//	cout << "Capture Image done" << endl;
				//Background_Subtraction(capture_num);
				
				ProcessImages(bgframe, frame, framecount);
				
				//Background_compare(frame);
				//}
		//}
	}  // End of while loop

	//play fg animation
	/*namedWindow("result");
	moveWindow("result", 0, frame.rows + 20);
	waitKey();
	counter = 0;
	Mat image;
	while (true)
	{
		stringstream ss;
		string filename;
		ss << "capture_mask\\mask_" << counter++ << ".jpg";
		ss >> filename;
		
		image = imread(filename);
		if (image.empty())
		{
			cout << "Can't read image from your computer" << endl;
			break;
		}
		ss.clear();
		
		imshow("result", image);
		waitKey(300);
		if (counter > 20)
			break;
	} */


	return 0;
}

