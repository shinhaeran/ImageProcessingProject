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
			if(y+dy>=0 && x+dy>=0){
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

			if (y + dy >= 0 && x + dy >= 0) {
				if (y == 0 && x == 0) {
				}
				else
					sum += img[y+dy][x+dx] * fnH[dy+z][dx+z];
			}
		}
	}
	  
	return sum;
}



void masking(int **img, int **img_out, int height, int width,int n) {
	float **fnH = FloatAlloc2(n, n);


	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fnH[i][j] = (float)1 / (n*n);
		}
	}

	float sum = 0.0;

	
	for (int y = 0; y < height - ((n - 1) / 2); y++) {
		for (int x = 0; x < width - ((n - 1) / 2); x++) {//256-4
			if (x - (n - 1) < 0 || y - (n - 1) < 0 || x + (n - 1) > width - 1 || y + (n - 1) > height - 1)
				img_out[y][x] = img[y][x];
			else {
				sum = ReadBlockMasking(x, y, n, fnH, img);
				img_out[y][x]= (int)sum;
			}

		}
	}
	//("test1", img_out, width, height);
	//ImageShow("test", img, width, height);

}


//low-pass filter:변화가 많은걸 지움-> 사진이 뭉개짐
//high:변화가 적은걸 지움->
void main() { //합성곱, h:n*n block에 1/3, -> h랑 img block이랑 각 인덱스끼리 곱함->"Masking"
	int width, height;
	int **img = ReadImage("LENA256_salt(noise_add).bmp", &width, &height);
	int **img_out1 = IntAlloc2(width, height);
	int **img_out2= IntAlloc2(width, height);

	masking(img, img_out1, height, width, 3);
	meanFiltering(img, img_out2, width, height, 3);
	ImageShow("test", img, width, height);
	ImageShow("test1", img_out1, width, height);
	ImageShow("test2", img_out2, width, height);

}



//0~ 258 :  밝기 사이즈
/*
1024 - 768
*/
