#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
void main()
{
	Mat img,img1, templ, result;
	img = imread("5zlotychA.jpg", IMREAD_COLOR);
	templ = imread("5zl_male.jpg", IMREAD_COLOR);
	
	img.copyTo(img1);

	double bestVal = -1, bestScale;
	Point bestLoc;
	for (double i = .5; i > .01; i -= .05)
	{
		resize(img, img1, Size(), i, i);
		int result_cols = img1.cols - templ.cols + 1;
		int result_rows = img1.rows - templ.rows + 1;
		result.create(result_rows, result_cols, CV_32FC1);
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		
		matchTemplate(img1, templ, result, TM_CCOEFF_NORMED);
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		cout << "Skala:" << i << " - " << maxVal << endl;
		if (maxVal > bestVal)
		{
			bestVal = maxVal;
			bestLoc = maxLoc;
			bestScale = i;
		}

	}
	cout << "drobny kroczek"<<endl;
	if (bestVal > 0.4)
		for (double i = bestScale + .05; i > bestScale - .05; i -= .005)
		{
			resize(img, img1, Size(), i, i);
			int result_cols = img1.cols - templ.cols + 1;
			int result_rows = img1.rows - templ.rows + 1;
			result.create(result_rows, result_cols, CV_32FC1);
			double minVal; double maxVal; Point minLoc; Point maxLoc;

			matchTemplate(img1, templ, result, TM_CCOEFF_NORMED);
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

			cout << "Skala:" << i << " - " << maxVal << endl;
			if (maxVal > bestVal)
			{
				bestVal = maxVal;
				bestLoc = maxLoc;
				bestScale = i;
			}
		}
	else
		cout << "za niski wspolczynnik dopasowania";

	resize(img, img1, Size(), bestScale, bestScale);
	rectangle(img1, bestLoc, Point(bestLoc.x + templ.cols, bestLoc.y + templ.rows), CV_RGB(255,0,0), 3);
	namedWindow("1", 0);
	imshow("1", img1);

	//imshow(result_window, result);
	imshow("template", templ);
	waitKey(0);
}