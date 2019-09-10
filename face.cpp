// face.cpp : Defines the entry point for the console application.
// Name: Georgios Ntamatis
// Student ID: 4340

#include "stdafx.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/svm_threaded.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <math.h>       /* sqrt */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::ml;

int false_ = 0;

Mat labels;

String face_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
String predictor_name = "shape_predictor_68_face_landmarks.dat";

CascadeClassifier face_cascade;

cv::Mat grayMat, colorMat;

std::vector<std::vector<double>> train_data;  /* Vector to store all training data. */
std::vector<int> train_data_labels;			  /* Vector to store all training labels. */

string source_path = "C:/Images/";      /* Path to folder containing all the source data. */
string emot_path = "C:/Images/exp.txt";
string svm_dat = "emotion_svm.dat"; /* SVM classifier name. */

std::string predict[7] = { "angry", "disgust", "fear", "happy", "neutral", "sad", "surprised" };

bool trained = false;

typedef matrix<double, 4556, 1> sample_type;
typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
typedef radial_basis_kernel<sample_type> rbf_kernel;


/*
Pase data to proper format for trainer.
*/
void generateData(std::vector<sample_type>& data, std::vector<double>& labels)
{
	for (int i = 0; i < train_data.size(); i++)
	{
		sample_type temp;
		for (int j = 0; j < train_data[0].size(); j++)
		{
			temp(j) = train_data[i][j];
		}
		data.push_back(temp);
		labels.push_back((double)train_data_labels[i]);
	}
}

/*
Calculate the length between two points.
*/
double length(point po1, point po2)
{
	int x1, y1, x2, y2;
	double dist;
	x1 = po1.x();
	y1 = po1.y();
	x2 = po2.x();
	y2 = po2.y();
	dist = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
	dist = sqrt(dist);
	return dist;
}

/*
Calculate the angle between two points.
*/
double angle(point a, point b)
{
	int x1, y1, x2, y2;
	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();
	if ((x1 - x2) == 0)
		if ((y1 - y2) > 0)
			return (pi / 2);
		else
			return (-pi / 2);
	else
		return atan(double(y1 - y2)) / (x1 - x2);
}

/*
Get a vector containing all classification feature in respect of the full_object_detection data.
*/
std::vector<double> getFeatureVector(dlib::full_object_detection shape)
{
	/*
	Getting only relative points of the set.
	For every pair of points on the shape object
	1st append the length of the points.
	2nd append the angle of the points (Values should be not NaN or Inf so normalized due to pi).
	*/
	std::vector<double> feature;
	for (int j = 0; j < 68; j++)
		for (int k = 0; k < j; k++)
		{
			feature.push_back(length(shape.part(j), shape.part(k)));
			feature.push_back(angle(shape.part(j), shape.part(k)));
		}
	return feature;
}

/*
Convert a cv::Rect object to a dlib::rectangle object.
Found from https://stackoverflow.com/questions/34871740/convert-opencvs-rect-to-dlibs-rectangle
*/
static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

bool exist(const std::string& name)
{
	ifstream file(name);
	if (!file)            // If the file was not found, then file is 0, i.e. !file=1 or true.
		return false;    // The file was not found.
	else                 // If the file was found, then file is non-0.
		return true;     // The file was found.
}

/*
Normalize the cv::Mat image.
Found from https://bytefish.de/dev/libfacerec/tutorial/facerec_gender_classification.html
*/
static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

int main(int argc, char** argv)
{
	char a;
	try
	{
		std::cout << ">> Initializing..." << endl;
		/* Load Cascade and Predictor Data. */
		std::cout << ">> Loading OpenCV cascades." << endl;
		if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading1\n"); return -1; };
		std::cout << ">> Loading Dlib cascades." << endl;
		shape_predictor sp;
		deserialize(predictor_name) >> sp;
		std::cout << ">> Cascades loaded." << endl;
		/* Read Training source Data. */
		/* Read all *.JPG Images on folder. */
		/* Read emotion label of each Images. */
		std::vector<String> image_sources;

		std::ifstream file(emot_path.c_str(), ifstream::in);
		if (!file)
		{
			string error_message = ">> No valid input file was given, please check the given filename.";
			CV_Error(CV_StsBadArg, error_message);
		}
		std::cout << ">> Loading source image data from (" << source_path << "* ." << endl;
		char separator = ',';
		string line, path, classlabel;
		int i = 0;
		while (getline(file, line)) {
			if (line.length() > 57)
				line = line.substr(0, 48) + line.substr(53, 60);
			stringstream liness(line);
			getline(liness, path, separator);
			getline(liness, classlabel);
			if (!classlabel.empty()) {
				image_sources.push_back(source_path + path);
				labels.push_back(atoi(classlabel.c_str()));//in labels the emotion of the images
			}
		}
		std::cout << ">> Image sources size is : " << image_sources.size() << std::endl;
		std::cout << ">> Training labels size is : " << labels.size() << "." << endl;
		if (labels.cols >= predict->size())
		{
			string error_message = ">> No valid input file was given, please check the given file.";
			return -1;
		}
		std::cout << ">> Initialized correctly." << endl;
		/* 4- Train SVM. */
		if (trained == false)
		{
			std::cout << ">> Do you want to train the model with the given source  : [Y/ N]" << endl;
			std::cin >> a;
			if (a == 'Y' || a == 'y')
			{
				/* Parse train source images and pass to svm. */
				for (int w = 0; w < image_sources.size(); w += 50)
				{
					/* Open cv image. */
					colorMat = imread(image_sources[w]);
					if (!colorMat.data)
					{
						std::cout << ">> Could not open or find the image file (" << image_sources[w] << ")." << std::endl;
						return -1;
					}
					Mat normMat = norm_0_255(colorMat);
					cv::cvtColor(normMat, grayMat, CV_RGB2GRAY); /* Convert to Grayscale. */
					equalizeHist(grayMat, grayMat);
					/* Convert to dlib object. */
					dlib::array2d<unsigned char> img_gray;
					dlib::assign_image(img_gray, dlib::cv_image<unsigned char>(grayMat));
					int ptr = labels.at<int>(w);
					String emot = predict[ptr];
					std::cout << ">> Parsing image file (" << image_sources[w] << ") with given state " << emot << "." << endl;
					/* Detect faces. */
					std::vector<Rect> faces;
					try
					{
						face_cascade.detectMultiScale(grayMat, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
						std::cout << ">> Face object(s) detected." << endl;
					}
					catch (exception ex)
					{
						std::cout << "exception" << endl;
					}
					/* Parse face. */
					if (faces.size() >= 1)
					{
						std::cout << ">> Analyzing face object." << endl;
						/* Covert to dlib rectangle. */
						dlib::rectangle dlib_rect = openCVRectToDlib(faces[0]);
						full_object_detection shape = sp(img_gray, dlib_rect);
						/* Parse valid face landmarks. */
						if (shape.num_parts() == 68)
						{
							std::cout << ">> Image landmarks extracted (" << shape.num_parts() << " landmarks)." << endl;
							std::vector<double> landmarks_vector = getFeatureVector(shape); /* Reshape data to fit the svm model. */
							std::cout << ">> Classification data analyzed. (" << landmarks_vector.size() << " features)." << endl;
							try
							{
								train_data.push_back(landmarks_vector);
								std::cout << ">> Classification data added to train data." << endl;
							}
							catch (Exception ex)
							{
								std::cout << ex.msg << endl;
							}
							try
							{
								train_data_labels.push_back(ptr);
							}
							catch (Exception ex)
							{
								std::cout << ex.msg << endl;
							}
						}
						else
						{
							std::cout << ">> Can not add state to traindata." << endl;
						}
					}
				}

				/* Train the SVM. */
				try
				{
					/* Parsing all train data. */
					std::vector<sample_type> svm_data;
					std::vector<double> labels;
					generateData(svm_data, labels);
					ovo_trainer trainer;
					svm_nu_trainer<rbf_kernel> rbf_trainer;
					/* Giving kernel proper params. */
					rbf_trainer.set_kernel(rbf_kernel(1.4641e-05));
					rbf_trainer.set_nu(0.0498789);
					trainer.set_trainer(rbf_trainer); /* Settings the kernel/ */
					randomize_samples(svm_data, labels); /* Randomizing the sample .*/
					one_vs_one_decision_function<ovo_trainer> df = trainer.train(svm_data, labels); /* Train the svml as One-vs-One.*/
					one_vs_one_decision_function<ovo_trainer, decision_function<rbf_kernel> > df2;  /* Computing the desision function .*/
					df2 = df;
					serialize(svm_dat) << df2; /* Save the trainer. */
				}
				catch (Exception ex)
				{
					std::cout << ex.msg << endl;
				}
			}
		}
		std::cout << "Do you want to test the model on the given source; [Y / N]" << endl;
		std::cin >> a;
		if (a == 'Y' || a == 'y')
		{
			/* Loading the svm. */
			one_vs_one_decision_function<ovo_trainer, decision_function<rbf_kernel>	> ep;
			deserialize(svm_dat) >> ep;
			/* Parse train source images and pass to svm. */
			for (int w = 0; w < image_sources.size(); w += 1)
			{
				/* Open cv image. */
				colorMat = imread(image_sources[w]);
				if (!colorMat.data)
				{
					std::cout << ">> Could not open or find the image file (" << image_sources[w] << ")." << std::endl;
					return -1;
				}
				Mat normMat = norm_0_255(colorMat);
				cv::cvtColor(normMat, grayMat, CV_RGB2GRAY); /* Convert to Grayscale. */
				equalizeHist(grayMat, grayMat);
				/* Convert to dlib object. */
				dlib::array2d<unsigned char> img_gray;
				dlib::assign_image(img_gray, dlib::cv_image<unsigned char>(grayMat));
				int ptr = labels.at<int>(w);
				/* Detect faces. */
				std::vector<Rect> faces;
				try
				{
					face_cascade.detectMultiScale(grayMat, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
					std::cout << ">> Face object(s) detected." << endl;
				}
				catch (exception ex)
				{
					std::cout << "exception" << endl;
				}
				/* Parse face on image. */
				for (int i = 0; i < faces.size(); i++)
				{
					std::cout << ">> Analyzing face object." << endl;
					/* Locate each face object. */
					int x = faces[i].x;
					int y = faces[i].y;
					/* Create rectangle. */
					cv::rectangle(colorMat, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8, 0);
					/* Covert to dlib rectangle. */
					dlib::rectangle dlib_rect = openCVRectToDlib(faces[i]);
					full_object_detection shape = sp(img_gray, dlib_rect);
					/* Parse face landmarks. */
					if (shape.num_parts() == 68)
					{
						/* Converting landmarks to cv points. */
						for (int k = 0; k < shape.num_parts(); k++)
						{
							cv::Point landmark(shape.part(k).x(), shape.part(k).y());
							if (k > 16) /* We do not need the face ear to ear points. */
							{
								cv::circle(colorMat, landmark, 1, Scalar(0, 255, 0), -1);
							}
							else
							{
								cv::circle(colorMat, landmark, 1, Scalar(0, 0, 255), -1);
							}
						}
						std::cout << ">> Image state extracted (" << shape.num_parts() << " landmarks)." << endl;
						std::vector<double> landmarks_vector = getFeatureVector(shape);
						sample_type temp;
						for (int i = 0; i < landmarks_vector.size(); i++)
						{
							temp(i) = landmarks_vector[i];
						}
						int num_of_label = ep(temp);
						String prediction = predict[num_of_label];
						std::cout << ">> Image(" << image_sources[w] << ") prediction is " << prediction << "." << endl;
						if (num_of_label != ptr)
						{
							String correct_prediction = predict[ptr];
							std::cout << ">> Prediction is false (should be " << correct_prediction << ")" << endl;
							false_++;
						}
						else
						{
							std::cout << ">> Prediction is correct" << endl;
						}
						cv::Point textOrg(x - 10, y - 10);
						cv::putText(colorMat, prediction, textOrg, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 0), 2, 8);
						cv::imshow("Image Source", colorMat);
						cv::waitKey(1);
					}
					else
					{
						std::cout << ">> Can not proccess image data." << endl;
					}
				}
			}
			/* Print test stats. */
			std::cout << "Correct Predictions : " << (image_sources.size() - false_) << endl;
			std::cout << "Failed  Predictions : " << false_ << endl;
			std::cout << "Correct Ratio      : " << (double)(image_sources.size() - false_) / (image_sources.size()) * 100 << "%" << endl;
			std::cout << "Failure Ratio      : " << (double)(false_) / (image_sources.size()) * 100 << "%" << endl;
		}
		std::cout << "Do you want to test the model on realtime : [Y / N]" << endl;
		std::cin >> a;
		if (a == 'Y' || a == 'y')
		{
			/* Loading the svm. */
			one_vs_one_decision_function<ovo_trainer, decision_function<rbf_kernel>	> ep;
			deserialize(svm_dat) >> ep;
			std::cout << ">> Initializing video capture device." << endl;
			cv::VideoCapture cap;
			cap.open(0); /* Open Device. */
			if (!cap.isOpened())
			{
				std::cout << ">> Unable to connect to the capture device." << endl;
				return -1;
			}
			char key = 0;
			for (;;)
			{
				cap >> colorMat;
				Mat original = colorMat.clone(); /* Clone the current frame. */
				Mat normMat = norm_0_255(original);
				cv::cvtColor(normMat, grayMat, CV_RGB2GRAY); /* Convert to Grayscale. */
				equalizeHist(grayMat, grayMat);
				/* Convert to dlib object. */
				dlib::array2d<unsigned char> img_gray;
				dlib::assign_image(img_gray, dlib::cv_image<unsigned char>(grayMat));
				/* Detect faces. */
				std::vector<Rect> faces;
				try
				{
					face_cascade.detectMultiScale(grayMat, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

				}
				catch (exception ex)
				{
					std::cout << "exception" << endl;
				}
				int x = 0;
				int y = 0;
				/* Parse face. */
				for (int i = 0; i < faces.size(); i++)
				{
					std::cout << ">> Face object(s) detected." << endl;
					/* Covert to dlib rectangle. */
					dlib::rectangle dlib_rect = openCVRectToDlib(faces[i]);
					full_object_detection shape = sp(img_gray, dlib_rect);
					x = faces[i].x;
					y = faces[i].y;
					/* Create rectangle. */
					cv::rectangle(original, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8, 0);
					/* Converting landmarks to cv points. */
					std::vector<double> landmarks_vector = getFeatureVector(shape);
					sample_type temp;
					for (int i = 0; i < landmarks_vector.size(); i++)
					{
						temp(i) = landmarks_vector[i];
					}
					for (int k = 0; k < shape.num_parts(); k++)
					{
						cv::Point landmark(shape.part(k).x(), shape.part(k).y());
						if (k > 16) /* We do not need the face ear to ear points. */
						{
							cv::circle(original, landmark, 1, Scalar(0, 255, 0), -1);
						}
						else
						{
							cv::circle(original, landmark, 1, Scalar(0, 0, 255), -1);
						}
					}
					int num_of_label = ep(temp);
					String prediction = predict[num_of_label];
					std::cout << ">> Prediction is " << prediction << "." << endl;
					cv::Point textOrg(x - 10, y - 10);
					cv::putText(original, prediction, textOrg, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 0), 2, 8);
				}
				cv::imshow("Video Source", original);
				char key = (char)cv::waitKey(20);
				if (key == 27)
					break;
			}
		}

	}
	catch (exception& e)
	{
		std::cout << "\nexception thrown!" << endl;
		std::cout << e.what() << endl;
	}
}

// End of File ----------------------------------------------------------------------------------------
