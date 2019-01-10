#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include<time.h>

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
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int width, int height)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int width, int height)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int width, int height)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(const char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(const char* winname, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(const char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
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
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(const char* winname, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
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



int CaculateBlockDiff_color(int x, int y, int_rgb** img, int_rgb** block, int bsize) {
	int_rgb error;
	error.r = 0; error.g = 0; error.b = 0;

	for (int Y = 0; Y < bsize; Y++) {
		for (int X = 0; X < bsize; X++) {
			error.r += abs(block[Y][X].r - img[Y + y][X + x].r);
			error.g += abs(block[Y][X].g - img[Y + y][X + x].g);
			error.b += abs(block[Y][X].b - img[Y + y][X + x].b);
		}
	}

	return (error.r + error.g + error.b);
}

void Compare_color(int x, int y, int_rgb** block_out[], int bsize, int_rgb** img, int_rgb** img_out) {
	int err, index_min = 0, err_min = 1000000;

	for (int i = 0; i < 510; i++) {
		err = CaculateBlockDiff_color(x, y, img, block_out[i], bsize); // block과 이미지의 차이가 가장 작은 값을 찾는다

		if (err < err_min) {
			err_min = err;
			index_min = i;
		}
	}
	 
	for (int Y = 0; Y < bsize; Y++) {
		for (int X = 0; X < bsize; X++) {
			img_out[Y + y][X + x] = block_out[index_min][Y][X];
		}
	}
}

void MultiTemplate_color(int height, int width, int_rgb** img, int_rgb** img_out, int_rgb** block_out[], int bsize) { // 여러 사진으로 템플릿 매칭하는 함수

	for (int y = 0; y < height; y += bsize) {
		for (int x = 0; x < width; x += bsize) {
			Compare_color(x, y, block_out, bsize, img, img_out);// 여러 개의 사진 중 가장 비슷한 값을 찾아서 그 자리에 집어넣는 함수
		}
	}
}

void rotation1(int_rgb** block_in, int_rgb** block_out, int bsize) { // +90회전
	for (int y = 0; y < bsize; y++) {
		for (int x = 0; x < bsize; x++) {
			block_out[x][bsize - 1 - y] = block_in[y][x];
		}
	}
}

void rotation2(int_rgb** block_in, int_rgb** block_out, int bsize) { // -90회전
	for (int y = 0; y < bsize; y++) {
		for (int x = 0; x < bsize; x++) {
			block_out[bsize - 1 - x][y] = block_in[y][x];
		}
	}
}

void rotation3(int_rgb** block_in, int_rgb** block_out, int bsize) { // 좌우대칭
	for (int y = 0; y < bsize; y++) {
		for (int x = 0; x < bsize; x++) {
			block_out[y][bsize - 1 - x] = block_in[y][x];
		}
	}
}

void rotation4(int_rgb** block_in, int_rgb** block_out, int bsize) { //  상하대칭
	for (int y = 0; y < bsize; y++) {
		for (int x = 0; x < bsize; x++) {
			block_out[bsize - 1 - y][x] = block_in[y][x];
		}
	}
}

#define NUM_T 510
void main1() { //컬러 회전 템플릿 매칭
	int height, width, h_height, h_width;
	int_rgb** img = ReadColorImage("Koala.jpg", &width, &height);
	int_rgb** img_out1 = IntColorAlloc2(width, height);
	int_rgb** img_out2 = IntColorAlloc2(width, height);
	char filename[100];
	int bsize = 32;
	int_rgb** block[NUM_T];
	int_rgb** block_out[NUM_T *5];

	
	for (int i = 0; i < NUM_T * 5; i++) {
		if (i < 510) {
			sprintf_s(filename, "dbs%04d.jpg", i);
			block[i] = ReadColorImage(filename, &h_width, &h_height);
		}
		block_out[i] = IntColorAlloc2(h_width, h_height);
	}

	MultiTemplate_color(height, width, img, img_out1, block, bsize);

	
	for (int y = 0; y < height; y += 32) {
		for (int x = 0; x < width; x += 32) {
			int err_out, index_min = 0, err_min = 1000000;

			for (int i = 0; i < NUM_T*5; i++) {
				if (i < NUM_T) {
					block_out[i] = block[i];
				}
				if (i >= NUM_T && i < NUM_T*2) {
					rotation1(block[i - 510], block_out[i], bsize);
				}
				if (i >= NUM_T*2 && i < NUM_T*3) {
					rotation2(block[i - 1020], block_out[i], bsize);
				}
				if (i >= NUM_T*3 && i < NUM_T*4) {
					rotation3(block[i - 1530], block_out[i], bsize);
				}
				if (i >= NUM_T*4 && i < NUM_T*5) {
					rotation4(block[i - 2040], block_out[i], bsize);
				}
				err_out = CaculateBlockDiff_color(x, y, img, block_out[i], bsize); // block과 이미지의 차이가 가장 작은 값을 찾는다

				if (err_out < err_min) {
					err_min = err_out;
					index_min = i;
				}
			}

			for (int Y = 0; Y < bsize; Y++) {
				for (int X = 0; X < bsize; X++) {
					img_out2[Y + y][X + x] = block_out[index_min][Y][X];
				}
			}
		}
	}

	ColorImageShow("input", img, width, height);
	ColorImageShow("output1", img_out1, width, height);
	ColorImageShow("output2", img_out2, width, height);
	waitKey(0);

}


int_rgb ReadBlock(int_rgb** block, int X, int Y, int size) { //siizee=2
	int_rgb sum;
	sum.r = 0; sum.g = 0; sum.b = 0;
	for (int dy = 0; dy < size; dy++) {
		for (int dx = 0; dx < size; dx++) {
			sum.r += block[Y+dy][X+dx].r;
			sum.g += block[Y+dy][X+dx].g;
			sum.b += block[Y+dy][X+dx].b;
		}		
	}

	return sum;
}


void main2_() { //32 -> 16 510개 잘 줄여졌나 확인

	int height, width;
	
	char filename[100];
	int bsize = 32;
	int b_out_size = 16;
	int_rgb** block[NUM_T];
	int_rgb** block_out[NUM_T];

	for (int i = 0; i < NUM_T; i++) {
		sprintf_s(filename, "dbs%04d.jpg", i);
		block[i] = ReadColorImage(filename, &width, &height);
		block_out[i] = IntColorAlloc2(b_out_size, b_out_size);
	}



	int_rgb** img = IntColorAlloc2(16 * 33, 16 * 16);

	int Y = 0; int X = 0;
	for (int i = 0; i < NUM_T; i++) {//NUM_T=510,
		
		int n = bsize / b_out_size; //2
		int_rgb  sum;

		for (int y = 0; y < bsize; y += n) {
			for (int x = 0; x < bsize; x += n) {
				sum = ReadBlock(block[i], x, y, n); //n=2
				block_out[i][int(y / n)][int(x / n)].r = sum.r / 4;
				block_out[i][int(y / n)][int(x / n)].g = sum.g / 4;
				block_out[i][int(y / n)][int(x / n)].b = sum.b / 4;
			}
		}

		for (int y = 0; y < 16; y++) {
			for (int x = 0; x < 16; x++) {
				img[Y + y][X + x] = block_out[i][y][x];
			}
		}
	
		if (X < NUM_T)
			X += 16;
		else {
			X = 0;
			Y += 16;
		}
		
	}

	

	ColorImageShow("input", img, 16*33, 16*16);

	waitKey(0);
}


void Reduce_Template(int_rgb*** block, int_rgb*** block_out, int bsize, int b_out_size) {
	int_rgb** img1 = IntColorAlloc2(16 * 33, 16 * 16);

	int Y = 0; int X = 0;
	for (int i = 0; i < NUM_T; i++) {//NUM_T=510,

		int n = bsize / b_out_size; //2
		int_rgb  sum;

		for (int y = 0; y < bsize; y += n) {
			for (int x = 0; x < bsize; x += n) {
				sum = ReadBlock(block[i], x, y, n); //n=2
				block_out[i][int(y / n)][int(x / n)].r = sum.r / (n*n);
				block_out[i][int(y / n)][int(x / n)].g = sum.g / (n*n);
				block_out[i][int(y / n)][int(x / n)].b = sum.b / (n*n);
			}
		}

		for (int y = 0; y < b_out_size; y++) {
			for (int x = 0; x < b_out_size; x++) {
				img1[Y + y][X + x] = block_out[i][y][x];
			}
		}

		if (X < NUM_T)
			X += b_out_size;
		else {
			X = 0;
			Y += b_out_size;
		}

	}
	//ColorImageShow("input1", img1, 16 * 33, 16 * 16);
}

void main_4() { //16 -> 8 510개 잘 줄여졌나 확인
	int bsize = 32;
	int b_out_size = 8;
	int height, width, h_height, h_width;
	int_rgb** img = ReadColorImage("Koala.jpg", &width, &height);
	int_rgb** img_out1 = IntColorAlloc2(width, height);
	int_rgb** img_out2 = IntColorAlloc2(width, height);
	char filename[100];

	int_rgb** block[NUM_T];
	int_rgb** block_out[NUM_T];


	for (int i = 0; i < NUM_T; i++) {
		sprintf_s(filename, "dbs%04d.jpg", i);
		block[i] = ReadColorImage(filename, &h_width, &h_height);
		block_out[i] = IntColorAlloc2(b_out_size, b_out_size);
	}



	int_rgb** img1 = IntColorAlloc2(16 * 33, 16 * 16);

	int Y = 0; int X = 0;
	for (int i = 0; i < NUM_T; i++) {//NUM_T=510,

		int n = bsize / b_out_size; //2
		int_rgb  sum;

		for (int y = 0; y < bsize; y += n) {
			for (int x = 0; x < bsize; x += n) {
				sum = ReadBlock(block[i], x, y, n); //n=2
				block_out[i][int(y / n)][int(x / n)].r = sum.r / (n*n);
				block_out[i][int(y / n)][int(x / n)].g = sum.g / (n*n);
				block_out[i][int(y / n)][int(x / n)].b = sum.b / (n*n);
			}
		}

		for (int y = 0; y < b_out_size; y++) {
			for (int x = 0; x < b_out_size; x++) {
				img1[Y + y][X + x] = block_out[i][y][x];
			}
		}

		if (X < NUM_T)
			X += 8;
		else {
			X = 0;
			Y += 8;
		}

	}
	bsize = 8;
	//MultiTemplate_color(height, width, img, img_out1, block, bsize);
	MultiTemplate_color(height, width, img, img_out1, block_out, bsize);


	ColorImageShow("input", img1, 16 * 33, 16 * 16);
	ColorImageShow("output1", img_out1, width, height);

	waitKey(0);
}


//일단 원래 이미지에서 random위치에 random크기만큼 이미지 뽑음
void ExtractRandom(int i, int randomX, int randomY, int_rgb** img_original, int_rgb** img_out, int_rgb** img_extract, int* extract_size) {   
	
	switch (i) //랜덤크기만큼 할당
	{
		case 1: //32px
			*extract_size = 32;
			break;
		case 2: //16px
			*extract_size = 16;
			break;
		case 3: //8px
			*extract_size = 8;
			break;
	}
	img_extract = IntColorAlloc2(*extract_size, *extract_size);
	
	randomY = MIN(768 - 51, randomY);
	//랜덤 위치에 이미지 뽑음
	for (int dy = 0; dy < *extract_size; dy++) {
		for (int dx = 0; dx < *extract_size; dx++) {
			img_extract[dy][dx] = img_original[randomY + dy][randomX + dx];
		}
	}

	//ColorImageShow("output1", img_extract, *extract_size, *extract_size);

	waitKey(0);
}


void ExtractTempleteMatching(int randomX, int randomY, int_rgb** img_original, int_rgb** img_out, int_rgb** img_extract, int extract_size, int height, int width,
	int_rgb*** block_out_32, int_rgb*** block_out_16, int_rgb*** block_out_8) { //그거로 템플릿 매칭

	switch (extract_size)
	{
		
	case 32:
		Compare_color(randomX, randomY, block_out_32, extract_size, img_original, img_out);
		break;

	case 16:
		Compare_color(randomX, randomY, block_out_16, extract_size, img_original, img_out);
		break;

	case 8:
		Compare_color(randomX, randomY, block_out_8, extract_size, img_original, img_out);
		break;
	}


	//중복하는지 검사

}


void main() {
	int bsize32 = 32; int bsize16 = 16; int bsize8 = 8;	

	int height, width;
	int_rgb** img_original = ReadColorImage("cat4.jpg", &width, &height);//원래 이미지
	int_rgb** img_out = IntColorAlloc2(width, height);
	int_rgb** img_extract=0;
	int_rgb** block_out_32[NUM_T]; int_rgb** block_out_16[NUM_T]; int_rgb** block_out_8[NUM_T];
	int extract_size = 0;
	char filename[100];
	
	//32, 16, 8px db 만들기
	for (int i = 0; i < NUM_T; i++) {

		sprintf_s(filename, "dbs%04d.jpg", i);
		block_out_32[i] = ReadColorImage(filename, &bsize32, &bsize32);
		block_out_16[i] = IntColorAlloc2(bsize16, bsize16);
		block_out_8[i] = IntColorAlloc2(bsize8, bsize8);
	}

	Reduce_Template(block_out_32, block_out_16, bsize32, bsize16);
	Reduce_Template(block_out_16, block_out_8 , bsize16, bsize8 );

	srand((unsigned)time(NULL));
	
	//랜덤 크기, 위치 설정
	int percentage[10] = { 1,1,2,2,2,2,2,3,3,3 }; // 1: 32px, 2:16px, 3:8px ->확률 정해놓기.
	int randomX = 0; int randomY = 0;

	int count = 0;
	
	for (int Y = 50; Y < height; Y+=15) {
		for (int X = 50; X < width; X+=15) {
			
			int i = percentage[rand()%10]; // rand()%10 : 0~9 무작위 수 -> percentage 배열의 index로 들어감->확률기반
			
			randomX = MIN( width-50 ,MAX(X - (rand()%50), 0) ); //x,y 랜덤 좌표는 해당 width~width-50, height~height-50
			randomY = MIN( height- 50, MAX(Y - (rand()%50), 0) );
			randomY = MIN(height - 50, randomY);

			//일단 원래 이미지에서 random위치에 random크기만큼 이미지 뽑음
			ExtractRandom(i, randomX, randomY, img_original, img_out, img_extract, &extract_size);
			//그거로 템플릿 매칭 
			ExtractTempleteMatching(randomX, randomY, img_original, img_out, img_extract, extract_size, height, width, block_out_32, block_out_16, block_out_8);		
		}
		
	}

	ColorImageShow("output1", img_out, width, height);

	waitKey(0);
}


//0~ 258 :  밝기 사이즈
/*
1024 - 768
*/



