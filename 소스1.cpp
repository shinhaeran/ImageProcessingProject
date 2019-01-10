#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

#define PI 3.14159265359
#define SQ(x) ((x)*(x))//SQ가 (x*x)면 -3 나옴: 1-2가 고대로 들어가니께
#define IMIN(x,y) ((x>y)?y:x)
#define IMAX(x,y) ((x<y)?y:x)

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int width, int height)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int width, int height)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int width, int height, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

void Prob0904() {
	int width, height;
	int **img;//2차원 배열의 위치를 저장하는 변수

	img = ReadImage("Penguins.jpg", &height, &width);
	ImageShow("test", img, height, width);

	IntFree2(img, height, width);
}

void Rectanguler(int **img, int width, int height, int delta) {

	for (int y = 192; y < 576; y++) {
		for (int x = 256; x < 768; x++) {
			img[y][x] = img[y][x] + delta;
			img[y][x] = IMAX(IMIN(img[y][x], 255), 0);
		}
	}
}
//255보다 커지면 0부터 커짐 ->범위제한
void Circle(int **img, int width, int height, int delta, int a, int b) { //밝기: 0~255 ->음수가 나오면 8비트->엄청 밝게 나옴
																		 //delta = (delta > 0) ? delta : 0;
																		 //delta = (delta > 255) ? 255 : delta;
	for (int y = 0; y < width; y++) {
		for (int x = 0; x < height; x++) {
			if ((x - a)*(x - a) + (y - b)*(y - b) < 1000) {
				//img[y][x] = (img[y][x] + delta > 255) ?  255 : img[y][x] + delta;
				img[y][x] = img[y][x] + delta;
				img[y][x] = IMAX(IMIN(img[y][x], 255), 0);

			}

		}
	}
}

void Prob0911() {
	//Prob0904();
	int width, height;
	int **img = ReadImage("Penguins.jpg", &width, &height);

	//y: 이미지의 세로 -x랑 y바꾸면 엄청 느려짐 ->  x방향 먼저 읽는게 빠르다.

	//Rectanguler(img, width, height,50);
	Circle(img, width, height, 50, 300, 300);

	ImageShow("test", img, width, height);



	IntFree2(img, height, width);

}

void PixelCount(int **img, int* histogram) {//int histogram[] 과 같다

											//int histogram[256] = { 0 };//이미지 안에 각 밝기가 몇개씩인지 
	for (int y = 0; y < 768; y++) {
		for (int x = 0; x < 1024; x++) {
			histogram[img[y][x]]++;
		}
	}
	/*
	for (int i = 0; i < 256; i++)
	printf("%d \n", histogram[i]);
	printf("%d, %d", *histogram, *(histogram + 1));
	*/
	/*
	int* address;
	address = histogram + 2;
	printf("\n %d %d", address - 1, address[-1]);// 된다
	printf("\n %d %d", *(histogram-1), histogram[-1]);//안된다
	*/
}

void prob0911() {
	int width, height;
	int **img = ReadImage("Penguins.jpg", &width, &height);
	int histogram[256] = { 0 };
	PixelCount(img, histogram);
}


void mappintImage(int **img, int** img_out, int width, int height, int* histogram) {

	PixelCount(img, histogram);

	float pdf[256], cdf[256];


	pdf[0] = (float)histogram[0] / (width*height);
	cdf[0] = pdf[0];
	for (int i = 1; i < 256; i++) {
		pdf[i] = (float)histogram[i] / (width*height);
		cdf[i] = (float)cdf[i - 1] + pdf[i];
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = (int)(cdf[img[y][x]] * 255);
			img_out[y][x] = IMAX(IMIN(img_out[y][x], 255), 0);
			//printf("%d \n",img_out[y][x]);
		}
	}
}

void prob0912() {
	int width, height;
	int **img = ReadImage("tulip_dark.bmp", &width, &height);
	int **img_out = IntAlloc2(width, height);
	int histogram[256] = { 0 };

	mappintImage(img, img_out, width, height, histogram);

	ImageShow("test", img_out, width, height);



}
#define f(m,x,a,fa) m*(x-a)+fa 
void stretching(int a, int b, int fa, int fb, int** img, int** img_out, int width, int height) {

	int m = ((float)fb - fa) / (b - a);



	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = f(m, img[y][x], a, fa);
			img_out[y][x] = IMAX(IMIN(img_out[y][x], 255), 0);
		}
	}
	ImageShow("test", img_out, width, height);

}

void prob0919() {
	int width, height;
	int **img = ReadImage("Penguins.jpg", &width, &height);
	int **img_out = IntAlloc2(width, height);
	int histogram[256] = { 0 };

	stretching(70, 150,100 , 200, img, img_out, width, height);
	ImageShow("test1", img, width, height);
}
/*
void meanFiltering(int **img,int** img_out,int width,int height) {//가장자리 9개 더해서 평균
	int sum = 0;
	for (int y = 0; y < height-1; y++) {
		for (int x = 0; x < width-1; x++) {
			if (y == 0 || x == 0) {
				img_out[y][x] = img[y][x];
			}
			else{
			sum = img[y - 1][x - 1] + img[y-1][x] + img[y - 1][x + 1]+ img[y][x - 1] + img[y][x + 1] + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1];
			img_out[y][x] = sum / 9;
			img_out[y][x] = IMAX(IMIN(img_out[y][x], 255), 0);
			}


		}
	}
	ImageShow("test1", img_out, width, height);
	ImageShow("test", img, width, height);

}
*/
void meanFiltering(int **img, int** img_out, int width, int height,int n) {//nxn
	int sum = 0;
	int z = n - 2;//3:1 , 5:2
	for (int y = z; y < height-z ; y++) {//3: 1~254
		for (int x =z; x < width-z ; x++) {

			for (int dy = -z; dy < (z + 1); dy++) {//3: -1~1 , 5:-2~2
				for (int dx = -z; dx < (z + 1); dx++) {
					if (y == 0 && x == 0) {
					}
					else sum += img[y + dy][x + dx];
				}
			}
			
			img_out[y][x] = sum / (n*n) + 0.5;
			img_out[y][x] = IMAX(IMIN(img_out[y][x], 255), 0);
			sum = 0;

		}
	}
	ImageShow("test1", img_out, width, height);
	ImageShow("test", img, width, height);

}


void ReadBlock(int x, int y, int n1, int n2, int *block,int** img) {
	int index = 0;
	int z = n1 - 2;
	


	for (int dy = -z; dy < (z + 1); dy++) {//3: -1~1 , 5:-2~2
		for (int dx = -z; dx < (z + 1); dx++) {
			if(y+dy>=0 && x+dx>=0){
				if (y == 0 && x == 0) {
				}
				else block[index++] = img[y+dy][x+dx];
			}
		}
	}
}

int Sorting(int *block, int n) {//버블정렬
	int temp = 0;
	for (int i = 0; i < n ; i++)
	{
		for (int j = 0; j < n - i; j++)
		{
			if (block[j] < block[j + 1])
			{
				temp = block[j];
				block[j] = block[j + 1];
				block[j + 1] = temp;
			}
		}
	}
	return block[n / 2];
}

void medianFilterNXN(int width, int height, int n1 ,int** img, int** img_out) {

	//meanFiltering(img, img_out, width, height,5);
	int i = 0;
	int* block = (int*)malloc(n1 * n1* sizeof(int));
	for (int y = 0; y < height - ((n1-1)/ 2); y++) {
		for (int x = 0; x < width - ((n1 - 1) / 2); x++) {//256-4
			if (x - (n1 - 1) < 0 || y - (n1 - 1) < 0 || x + (n1 - 1) > width - 1 || y + (n1 - 1) > height - 1)
				img_out[y][x] = img[y][x];
			else {
				ReadBlock(x, y, n1, n1, block, img);// 값을 읽어와서  block배열에 저장
													//printf("%d %d\n", x, y);
				img_out[y][x] = Sorting(block, n1*n1);//중간값 찾아서 return 값을 block [4]로
			}

		}
	}

	ImageShow("test1", img_out, width, height);
	ImageShow("test", img, width, height);


}


void prob1004() {
	int width, height;
	int **img = ReadImage("LENA256_salt(noise_add).bmp", &width, &height);
	int **img_out = IntAlloc2(width, height);

	//meanFiltering(img, img_out, width, height,5);
	medianFilterNXN(width,height,11,img,img_out);

	
}


float** FloatAlloc2(int height, int width)

{
	float** tmp;  tmp = (float**)calloc(height, sizeof(float*));

	for (int i = 0; i<height; i++)

		tmp[i] = (float*)calloc(width, sizeof(float));

	return(tmp);

}

void FloatFree2(float** image, int height, int width)

{

	for (int i = 0; i<height; i++)

		free(image[i]);

	free(image);

}
float ReadBlockMasking(int x, int y, int n1, float **fnH, int** img) {
	int index = 0;
	int z = (n1 - 1)/2;
	float sum = 0.0;

	
	for (int dy = -z; dy <= z; dy++) {//3: -1~1 , 5:-2~2
		for (int dx = -z; dx <= z; dx++) {
			
					sum += img[y+dy][x+dx] * fnH[dy+z][dx+z];

		}
	}
	  
	return sum;
}



void masking1(int **img, int **img_out, int height, int width,int n) { // 가로 -1 -1 -1 ;gy 성분
	float **fnH1 = FloatAlloc2(n, n);

	int a = -1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fnH1[j][i] = a++;
		}
		a = -1;
	}

	


	float sum1 = 0.0;
	float sum2 = 0.0;

	
	for (int y = 0; y < height - ((n - 1) / 2); y++) {
		for (int x = 0; x < width - ((n - 1) / 2); x++) {//256-4
			if (x - (n - 1) < 0 || y - (n - 1) < 0 || x + (n - 1) > width - 1 || y + (n - 1) > height - 1)
				img_out[y][x] = img[y][x];
			else {
				sum1 = ReadBlockMasking(x, y, n, fnH1, img);
				
				//img_out[y][x]= fabs((int)sum1);
				img_out[y][x] = ((int)sum1);
			}
		}
	}
}
void masking2(int **img, int **img_out, int height, int width, int n) { // 가로 -1 0 1 ;  gx성분
	float **fnH1 = FloatAlloc2(n, n);

	int a = -1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fnH1[i][j] = a++;
		}
		a = -1;
	}
	/*
	for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	printf("%f ", fnH1[i][j]);
	}
	printf("\n");
	*/
	
	float sum1 = 0.0;

	for (int y = 0; y < height - ((n - 1) / 2); y++) {
		for (int x = 0; x < width - ((n - 1) / 2); x++) {//256-4
			if (x - (n - 1) < 0 || y - (n - 1) < 0 || x + (n - 1) > width - 1 || y + (n - 1) > height - 1)
				img_out[y][x] = img[y][x];
			else {
				sum1 = ReadBlockMasking(x, y, n, fnH1, img);
			
				//img_out[y][x] = fabs((int)sum1);
				img_out[y][x] = ((int)sum1);
			}

		}
	}
}


//low-pass filter:변화가 많은걸 지움-> 사진이 뭉개짐
//high:변화가 적은걸 지움->
void prob1009() { //합성곱, h:n*n block에 1/3, -> h랑 img block이랑 각 인덱스끼리 곱함->"Masking"
	int width, height;
	int **img = ReadImage("LENA256_salt(noise_add).bmp", &width, &height);
	int **img_out1 = IntAlloc2(width, height);
	int **img_out2= IntAlloc2(width, height);

	//masking(img, img_out1, height, width, 3);
	
	meanFiltering(img, img_out2, width, height, 3);
	ImageShow("test", img, width, height);
	ImageShow("test1", img_out1, width, height);//masking
	ImageShow("test2", img_out2, width, height);

}

int FindMax(int **img, int height, int width) {
	int max = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			max = IMAX(img[y][x], max);
		}
	}
	return max;
}

void Scaling(float alpha, int **img_out, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = alpha*img_out[y][x];//최댓값->255, 0->0
		}
	}
}


void prob1017()
{
	int width, height;
	int **img = ReadImage("lena512_gaussian.bmp", &width, &height);
	int **img_out1 = IntAlloc2(width, height);
	int **img_out2 = IntAlloc2(width, height);
	int **img_out3 = IntAlloc2(width, height);

	// 필터링하는 프로그램/함수
	masking1(img, img_out1, height, width, 3);
	masking2(img, img_out2, height, width, 3);

	int maxvalue1 = FindMax(img_out1, height, width);
	int maxvalue2 = FindMax(img_out2, height, width);

	float alpha1 = 255.0 / maxvalue1;
	float alpha2 = 255.0 / maxvalue2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out3[y][x] = img_out1[y][x]+ img_out2[y][x];
			//img_out3[y][x] = IMAX(IMIN(img_out3[y][x], 255), 0);
		}
	}
	int maxvalue3 = FindMax(img_out3, height, width);
	float alpha3 = 255.0 / maxvalue3;

	Scaling(alpha3, img_out3, height, width);
	Scaling(alpha1, img_out1, height, width);
	Scaling(alpha2, img_out2, height, width);

	ImageShow("test", img, width, height);
	ImageShow("test1", img_out1, width, height);
	ImageShow("test2", img_out2, width, height);
	ImageShow("test3", img_out3, width, height);
}

void FindEdgeAngle(int width,int height,int **img,int **img_out){
	
	float **theta = FloatAlloc2(width, height);
	int **gy = IntAlloc2(width, height);
	int **gx = IntAlloc2(width, height);

	masking1(img, gy, height, width, 3);//gy
	masking2(img, gx, height, width, 3);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			theta[y][x] = atan2((double)gy[y][x], gx[y][x]);

		}
	}
	//기울기: 255/(PI*2)
	float gradient = 255 / (PI * 2);

	for (int y = 0; y < height; y++)//직선에 대입
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = (int) (gradient *(theta[y][x]+PI));
		}
	}

	ImageShow("test", img, width, height);
	ImageShow("test1", img_out, width, height);

}
//int **gx,int **gy,float **theta,
void prob1024() {
	int width, height;
	int **img = ReadImage("lena512_gaussian.bmp", &width, &height);
	
	int **img_out = IntAlloc2(width, height);

	FindEdgeAngle(width, height, img, img_out);
}

int Interpolation(float x, float y, int ** img, int height,int width) {//100.3 201

	float deltaX = x - (int)x;
	float deltaY = y - (int)y;
	
	int Y = (1 - deltaX)*(1 - deltaY)*img[(int)y][(int)x]
		+ (deltaX)*(1 - deltaY)*img[(int)y][(int)x+1]
		+ (1 - deltaX)*(deltaY)*img[(int)y+1][(int)x]
		+ (deltaX)*(deltaY)*img[(int)y+1][(int)x + 1];

	return Y;

}
void InverseMatrix(float **M, float **M_1) {
	float a = M[0][0];
	float b = M[0][1];
	float c = M[1][0];
	float d = M[1][1];

	float Det = a*d - b*c;
	M_1[0][0] = d / Det;
	M_1[0][1] = -b / Det;
	M_1[1][0] = -c / Det;
	M_1[1][1] = a / Det;
}

void magnification(float m,int **img,int** img_out,int height,int width) {// m배율 하는거
	//m이 2면->2배니까 x사이에 2개 , y사이에 2개
	float** affineT = FloatAlloc2(2, 2);
	float** inv_affineT = FloatAlloc2(2, 2);

	affineT[0][0] = m; affineT[0][1] = 0;
	affineT[1][0] = 0; affineT[1][1] = m;

	InverseMatrix(affineT, inv_affineT);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0; // 그렇지 않으면 사진위에 회전된 사진이 나옴
							   // 회전시키는 값(좌표회전), 정수를 회전시켜 실수를 얻음->배열을 곱해준 것임..

			float newX = inv_affineT[0][0] *x + inv_affineT[0][1] * y;
			float newY = inv_affineT[1][0] *x + inv_affineT[1][1] * y;

			if (newX >= 0 && newY >= 0 && newX < width - 1 && newY < height - 1) { // 이미지의 크기 안에서만 표현
				img_out[y][x] = Interpolation(newX, newY, img, height, width); // 같은 값으로 함수에서 사용해서 모두 같은 값이 들어감

			}
			//else // 가장자리는 제외
			//	img_out[y][x] = img[y][x];
		}
	}
}

void rotateInterpolation(int height, int width, int** img_out, float radian, int** img) {

	float** affineT = FloatAlloc2(2, 2);
	float** inv_affineT = FloatAlloc2(2, 2);

	affineT[0][0] = cos(radian); affineT[0][1] = -sin(radian);
	affineT[1][0] = sin(radian); affineT[1][1] = cos(radian);

	InverseMatrix(affineT, inv_affineT);


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0; // 그렇지 않으면 사진위에 회전된 사진이 나옴
							   // 회전시키는 값(좌표회전), 정수를 회전시켜 실수를 얻음->배열을 곱해준 것임..

			float newX = inv_affineT[0][0]*x + inv_affineT[0][1] *y;
			float newY = inv_affineT[1][0] *x + inv_affineT[1][1]*y;

			if (newX >= 0 && newY >= 0 && newX < width && newY < height) { // 이미지의 크기 안에서만 표현
				img_out[y][x] = Interpolation(newX, newY, img, height, width); // 같은 값으로 함수에서 사용해서 모두 같은 값이 들어감
			}

			//else // 가장자리는 제외
			//	img_out[y][x] = img[y][x];
		}
	}
}
void centerRotate(int height, int width, int** img_out, float radian, int** img) {
	float centerX = width / 2.0;
	float centerY = height / 2.0;

	float** affineT = FloatAlloc2(2, 2);
	float** inv_affineT = FloatAlloc2(2, 2);

	affineT[0][0] = cos(radian); affineT[0][1] = -sin(radian);
	affineT[1][0] = sin(radian); affineT[1][1] = cos(radian);

	InverseMatrix(affineT, inv_affineT);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0; // 그렇지 않으면 사진위에 회전된 사진이 나옴
							   // 회전시키는 값(좌표회전), 정수를 회전시켜 실수를 얻음->배열을 곱해준 것임..


			float newX = inv_affineT[0][0] * (x - centerX) + inv_affineT[0][1] * (y - centerY) + centerX;
			float newY = inv_affineT[1][0] * (x - centerX) + inv_affineT[1][1] * (y - centerY) + centerY;

			if (newX >= 0 && newY >= 0 && newX < width-1 && newY < height-1) { // 이미지의 크기 안에서만 표현
				img_out[y][x] = Interpolation(newX, newY, img, height, width); // 같은 값으로 함수에서 사용해서 모두 같은 값이 들어감

			}
			//else // 가장자리는 제외
			//	img_out[y][x] = img[y][x];
		}
	}
}


void AffineTransform(int height, int width, int** img_out, float radian, int** img) {
	float centerX = width / 2.0;
	float centerY = height / 2.0;

	float** affineT = FloatAlloc2(2, 2);
	float** inv_affineT = FloatAlloc2(2, 2);

	affineT[0][0] = 0.5; affineT[0][1] = 1;
	affineT[1][0] = 1; affineT[1][1] = 0.8;

	InverseMatrix(affineT, inv_affineT);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0; 


			float newX = inv_affineT[0][0] * (x - centerX) + inv_affineT[0][1] * (y - centerY) + centerX;
			float newY = inv_affineT[1][0] * (x - centerX) + inv_affineT[1][1] * (y - centerY) + centerY;

			if (newX >= 0 && newY >= 0 && newX < width - 1 && newY < height - 1) { 
				img_out[y][x] = Interpolation(newX, newY, img, height, width); 
			}
		}
	}
}

void prob1106() {
	int width, height;
	int** img = ReadImage("lena512_gaussian.bmp", &height, &width);
	int** img_out1 = IntAlloc2(height, width);
	int** img_out2 = IntAlloc2(height, width);
	int** img_out3 = IntAlloc2(height, width);
	float radian;
	
	radian = 30;

	radian = radian / 180 * PI; 


	//magnification(2, img, img_out1, height, width);
	//centerRotate(height, width, img_out1, radian, img);
	AffineTransform(height, width, img_out1, radian, img);//평행사변형으로 만들면서 회전하는 것처럼 보이기

	ImageShow("input", img, height, width);
	ImageShow("output1", img_out1, height, width);
	//ImageShow("output2", img_out3, height, width);

	IntFree2(img, height, width);
	IntFree2(img_out1, height, width);
	IntFree2(img_out2, height, width);
	IntFree2(img_out3, height, width);

}
int ReadBlock2(int x, int y, int n1, int n2, int *block, int** img,int width,int height) {
	
	//int z = n1 - 2;
	int aa = 0;

	int indexB = 0;
	for (int dy = 0; dy < n1; dy++) {//3: -1~1 , 5:-2~2
		for (int dx = 0; dx < n1; dx++) {
			
			if (y + dy >= 0 && x + dx >= 0&& y + dy < height && x + dx < width) {
				
				block[indexB++] = img[y + dy][x + dx];
			}
			else return -1;
		}
	}

	return 0;
}

void TemplaeMatching(int **block,int bSize,int **img,int height, int width,int* x_out,int* y_out,int* terror) {

	int* imgBlock = (int*)malloc(bSize * bSize * sizeof(int));
	
	int a = 0;


	//ReadBlock()
	for (int y = 0; y < height; y=y+32) {
		for (int x = 0; x < width; x=x+32) {

			int result = ReadBlock2(x, y, bSize, bSize, imgBlock, img, width, height);
			int index = 0;
			int temp = 0;

			if (result == 0) {
				for (int dy = 0; dy < bSize; dy++) {
					for (int dx = 0; dx < bSize; dx++) {
						temp = temp+ abs(imgBlock[index++] - block[dy][dx]);//둘이 뺀거를 절대치 취함

					}
				}

				if (a==0)
					*terror = temp;

				if (temp < *terror) {
					*x_out = x;
					*y_out = y;
					*terror = temp;
				}
			}
			a++;
		}
	}
	
}
void drawBox(int **img,int x_out,int y_out,int width,int height) {
	for (int dy = 0; dy < 16; dy++) {
		for (int dx = 0; dx < 16; dx++) {
			//img[y_out + dy][x_out + dx]=0;
			if (dy == 0 || dx == 0 || dy == 15 || dx == 15)
				img[y_out + dy][x_out + dx] = 254;
		}
	}
	ImageShow("input", img, width, height);
}
void prob1113() {
	int width, height;
	int** img = ReadImage("koala.bmp", &width, &height);
	int** img_out = IntAlloc2(width, height);
	

	int x_out = 0;
	int y_out = 0;
	int terror = 0;

	int widthB, heightB;
	int** block = ReadImage("template.bmp", &heightB , &widthB);

	TemplaeMatching(block, 16, img, height, width, &x_out, &y_out, &terror);
	
	ImageShow("input", img, width, height);
	drawBox(img, x_out, y_out, width, height);

}


void ReadBlock_img1(int **block, int** img, int width, int height) {//+90도->상하대칭됨;
	int indexB = 0;
	

	for (int i = 0; i < width; i++) {
		
		for (int j = height - 1; j > 0; j--) {
			block[i][height-j] = img[j][i];
			//block[j][i] = img[height - j][width - i];
			
			
		}
	}

}

void ReadBlock_img2(int **block, int** img, int width, int height) {//-90도-> 어 좌우대칭됨;
	int indexB = 0;

	for (int i = width; i > 0; i--) {
		for (int j = 0; j <height; j++) {
			block[height-i][j] = img[j][i];
			//[j][i] = img[j][width - i];
		}
	}

}

void ReadBlock_img3(int **block, int** img, int width, int height) {//좌우대칭
	int indexB = 0;

	for (int i = width - 1; i >= 0; i--) {
		for (int j = 0; j <height; j++) {
			block[j][width - i] = img[j][i];
			//[j][i] = img[j][width - i];
		}
	}

}

void ReadBlock_img4(int **block, int** img, int width, int height) {//상하대칭(ㅇ)
	int indexB = 0;

	for (int i = height-1; i > 0; i--) {
		for (int j = 0; j < width; j++) {
			block[height - i][j] = img[i][j];
		}
	}

}


void prob1120() {
	int width, height;
	int widthB, heightB;
	int** img = ReadImage("koala.bmp", &width, &height);

	int** blocks[5];
	blocks[0]= ReadImage("template(flipping).bmp", &widthB, &heightB);
	for (int i = 1; i < 5; i++){
		blocks[i] = (int**)IntAlloc2(widthB, heightB);
	}


	ReadBlock_img1(blocks[1], blocks[0], widthB, heightB);
	ReadBlock_img2(blocks[2], blocks[0], widthB, heightB);
	ReadBlock_img3(blocks[3], blocks[0], widthB, heightB);
	ReadBlock_img4(blocks[4], blocks[0], widthB, heightB);


	//ImageShow("input0", img0, widthB, heightB);
	//ImageShow("input1", img1, widthB, heightB);
	//ImageShow("input2", img2, widthB, heightB);
	//ImageShow("input3", img3, widthB, heightB);
	//ImageShow("input4", img4, widthB, heightB);

	int x_out[4] = { 0};
	int y_out[4] = { 0 };
	int terror[4] = { 0 };
	

	for(int i=0;i<4;i++)
		TemplaeMatching(blocks[i+1], 16, img, height, width, &x_out[i], &y_out[i], &terror[i]);

	
	int index = 0;
	for (int i = 1; i < 4; i++) {
		if (terror[index] > terror[i])
			index = i;
	}
	
	
	drawBox(img, x_out[index], y_out[index], width, height);

}
void drawBlock(int **img,int **block, int x_out, int y_out) {
	for (int dy = 0; dy < 32; dy++) {
		for (int dx = 0; dx < 32; dx++) {
			//img[y_out + dy][x_out + dx]=0;
			img[y_out + dy][x_out + dx] =block[dy][dx] ;
		}
	}
	//ImageShow("input", img, width, height);
}


void prob() {
	int **block[510];
	int width, height;
	int widthB, heightB;
	//block = (int**)malloc(510 * sizeof(int**));
	char filename[100];
	for (int i = 0; i < 510; i++) {
		sprintf(filename, "dbs%04d.jpg", i);
		block[i] = ReadImage(filename,&widthB,&heightB);
	}
	int** img = ReadImage("koala.bmp", &width, &height);


	for(int i=0;i<510;i++){
		int x_out = 0 ;
		int y_out= 0;
		int terror=0;
		
		TemplaeMatching(block[i], 32, img, height, width, &x_out, &y_out, &terror);

		int index = 0;

		drawBlock(img, block[i], x_out, y_out);
		

	}
	ImageShow("aa", img, width, height);

	
}
void DrawSquare(int **block, int x1, int x2, int y1, int y2, int height, int  width) { //사각형 그리는 함수

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j >= x1 && j <= x2){
				block[y1][j] = 255;
				block[y2][j] = 255;
			}
			if (i >= y1 && i <= y2){
				block[i][x1] = 255;
				block[i][x2] = 255;
			}
		}
	}

}

void FindMaxMin(int **block, int *x1, int *x2, int *y1, int *y2,int height, int  width) { //사각형 그리려고 높이 너비 구하는거
	int stackX = 0; int stackY = 0;

	int tempX1 = width; int tempX2 = 0; int tempY1 = height; int tempY2 = 0;

	int i = 0; int j = 0;
	
	for (i = 0; i < height-1; i++) { //y
		for (j = 0; j < width; j++) { //x
			if (stackX==0&& block[i][j] >0) { //맨처음 0이었다가 흰색 등장
				stackX++; //그 라인 흰색등장표시
				tempX1 = IMIN(tempX1, j); 
			}
			
			if (stackX == 1 && block[i][j] == 0) {//흰색 등장 후 지금 검은색 등장
				tempX2 = IMAX(tempX2, j);
				stackX = 0; //라인 초기화
			}


			if (stackY == 0 && block[i][j] > 0) { //세로로 흰색 등장 
				stackY++;
				tempY1 = i; //그때 값이  y1
			}

			if (stackY == 1 && block[i][j] >0 && block[i+1][j]==0) { //흰색 등장 후 지금 세로에 검은색 등장
				tempY2 = IMAX(tempY2, i);
				//stackY = 10;
			}
		}  
	}

	*x1 = tempX1;
	*x2 = tempX2;
	*y1 = tempY1;
	*y2 = tempY2;
	
}

void DrawLine(int **block, int x1, int x2,int y1, int y2, int height, int  width) {
	float m = (float)(y1 - y2) / (x1 - x2);
	float a = m;
	float b = -1;
	float c = (-m*x1) + y1;
	float d = 0;

	int maxY = IMAX(y1, y2);
	int minY = IMIN(y1, y2);


	for (int y = minY; y < maxY; y++) {
		for (int x = 0; x < width; x++) {
			d = ((a*x) + (b*y) + c ) / sqrt(a*a + b*b);
			
			if (abs(d) < 1)
				block[y][x] = 255;
		}
	}
}



void extractionNUM(int **block, int height, int  width, int* x1, int* x2, int* y1, int* y2) { //숫자를 추출하는 함수

	int stack = 0;
	
	int offset[5]={0};

	int sumX = 0; int sumY = 0;

	int n = 0;
	
	for (int x= 0; x < width; x++) { 

		for (int y = 0; y < height; y++) { //세로 라인별로 sum값, 0이 나오면? 숫자 자르기
			sumY += block[y][x];
		}

		if (sumY == 0) {
			stack++;
		}

		if (stack > 60) { //공백이 60px넘을 때, 숫자를 자른다
			stack = 0;// 공백stack 초기화
			offset[n] = x; //offset
			n++;
		}
		sumY = 0;
	}

	/*x범위 대강 정해주기*/
	x1[0] = 0;
	x2[0] = offset[0] - 60;
	for (int i = 1; i < 4; i++) {
		x2[i] = offset[i] - 60;
		x1[i] = x2[i - 1];
	}

	

}

void FindMaxMin2(int **block, int *x1, int *x2, int *y1, int *y2, int height, int  width) { //정확한 x,y 범위 구하기
	int stackX = 0; int stackY = 0;

	int tempX1 = width; int tempX2 = 0; int tempY1 = height; int tempY2 = 0;

	int i = 0; int j = 0;

	for (i = *y1; i <  *y2 - 1; i++) { //y
		for (j = *x1; j < *x2; j++) { //x
			if (stackX == 0 && block[i][j] >0) { //맨처음 0이었다가 흰색 등장
				stackX++; //그 라인 흰색등장표시
				tempX1 = IMIN(tempX1, j);
			}

			if (stackX == 1 && block[i][j] == 0) {//흰색 등장 후 지금 검은색 등장
				tempX2 = IMAX(tempX2, j);
				stackX = 0; //라인 초기화
			}


			if (stackY == 0 && block[i][j] > 0) { //세로로 흰색 등장 
				stackY++;
				tempY1 = i; //그때 값이  y1
			}

			if (stackY == 1 && block[i][j] >0 && block[i + 1][j] == 0) { //흰색 등장 후 지금 세로에 검은색 등장
				tempY2 = IMAX(tempY2, i);
				//stackY = 10;
			}
		}
	}

	*x1 = tempX1;
	*x2 = tempX2;
	*y1 = tempY1;
	*y2 = tempY2;

}




void main() {//실기시험
	
	int** block;
	int width, height;

	block = ReadImage("num_img(0-4).bmp", &width, &height);


	int x1[5] = { 0 }; int x2[5] = { 0 }; int y1[5] = { 0 }; int y2[5] = { 0 };

	extractionNUM(block, height, width, x1,x2,y1,y2);

	/*대강 숫자 자른거 정확한 x,y범위 정해주기*/
	for (int i = 0; i < 4; i++) {
		FindMaxMin(block, &x1[i], &x2[i], &y1[i], &y2[i], height, width); //정확한 범위 잡아주고
		DrawSquare(block, x1[i], x2[i], y1[i], y2[i], height, width);//사각형 그려주기
		

		ImageShow("aa", block, width, height);//악 왜안돼ㅜㅜㅜㅜㅜ........
	}
	

	//ImageShow("aa", block, width, height);

	
	//int **block[10];
	//int width, height;
	//char filename[100];
	//
	//int x1 = 0; int x2 = 0; int y1 = 0; int y2 = 0;

	//for (int i = 0; i < 10; i++) { 
	//	sprintf(filename, "%d_org.bmp", i);
	//	block[i] = ReadImage(filename, &width, &height);//0~9 block 읽고 block[i]에 저장

	//	FindMaxMin(block[i], &x1, &x2, &y1, &y2, height, width);//사각형 그릴려고 사이즈 찾기
	//	DrawSquare(block[i], x1,x2, y1, y2, height, width); //사각형 그리기

	//	DrawLine(block[i], x1, x2, y1, y2, height, width);
	//	DrawLine(block[i], x2, x1, y1, y2, height, width);

	//	ImageShow("aa", block[i], width, height);
	//}





	

}




//0~ 258 :  밝기 사이즈
/*
1024 - 768
*/

