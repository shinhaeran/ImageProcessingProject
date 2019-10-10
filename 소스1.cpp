#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  
#include <time.h>
using namespace cv;
#define PI 3.14159265359
int** prepared_Avg16x16;
int** prepared_Avg8x8;
int** prepared_Avg4x4;
typedef struct {
	int r, g, b;
}int_rgb;
int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));

	return(tmp);
}
void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}
int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}
void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

int** ReadImage(const char* name, int* height, int* width)
{

	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);
	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);
	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{

	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imwrite(name, img);
}

void ImageShow(const char* winname, int** image, int height, int width)
{

	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}

int_rgb** ReadColorImage(char* name, int* height, int* width)
{

	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);
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

void WriteColorImage(char* name, int_rgb** image, int height, int width)
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

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
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

void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)

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

float** FloatAlloc2(int height, int width)
{

	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{

	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

typedef struct ERROR {
	int x;
	int y;
	int avg;
	int trans_version; //geometry_transform_version
	float a; //alpha
};

/*
geometric transform
0. A[y][x] = B[y][x]
1. A[y][x] = B[N-1-y][x]
2. A[y][x] = B[y][N-1-x]
3. A[y][x] = B[N-1-y][[N-1-x]

4. A[y][x] = B[x][y]
5. A[y][x] = B[N-1-x][y]
6. A[y][x] = B[x][N-1-y]
7. A[y][x] = B[N-1-x][n-1-y
*/

void geometric_transform0(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[y][x];
	}
}


void geometric_transform1(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[n - 1 - y][x];
	}
}

void geometric_transform2(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[y][n - 1 - x];
	}
}

void geometric_transform3(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[n - 1 - y][n - 1 - x];
	}
}


void geometric_transform4(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[x][y];
	}
}


void geometric_transform5(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[n - 1 - x][y];
	}
}

void geometric_transform6(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[x][n - 1 - y];
	}
}

void geometric_transform7(int** src, int n, int** des) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++)
			des[y][x] = src[n - 1 - x][n - 1 - y];
	}
}
void geometric_transform(int **src, int size, int**des, int version) {
	switch (version)
	{
	case 0: {
		geometric_transform0(src, size, des); break;
	}

	case 1: {
		geometric_transform1(src, size, des); break;
	}

	case 2: {
		geometric_transform2(src, size, des); break;
	}

	case 3: {
		geometric_transform3(src, size, des); break;
	}

	case 4: {
		geometric_transform4(src, size, des); break;
	}
	case 5: {
		geometric_transform5(src, size, des); break;
	}

	case 6: {
		geometric_transform6(src, size, des); break;
	}
	case 7: {
		geometric_transform7(src, size, des); break;
	}
	}

}

int getBlockAvg(int **image, int y, int x, int N) {
	int avg = 0;

	for (int j = y; j < y + N; j++) {
		for (int i = x; i < x + N; i++) {
			if (j > 255 || i > 255) avg += 255;
			else avg += image[j][i];
		}
	}

	return (avg / (N*N));
}

void downSize2(int **image, int **img_out, int N, int height, int width) {

	//int N = 8;
	int avg = 0;
	for (int y = 0; y < height; y += N) {
		for (int x = 0; x < width; x += N) {
			avg = getBlockAvg(image, y, x, N);
			img_out[y / N][x / N] = avg;
		}
	}


}


void RemoveMean(int** Block, int N, int** block_mean, int avg) {// 평균을 제거하는 함수-> block입력이 되면 평균 계산해서 원래 밝기에서 평균을 뺌->그게  block_mean

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			block_mean[j][i] = Block[j][i] - avg;
			//block_mean[j][i] = MAX(0, block_mean[j][i]);
			//block_mean[j][i] = MIN(255, block_mean[j][i]);
		}
	}

}
void RemoveMean_alpha(int** Block, int N, int** block_mean, float alpha) {// 평균을 제거하는 함수-> block입력이 되면 평균 계산해서 원래 밝기에서 평균을 뺌->그게  block_mean
	int avg = getBlockAvg(Block, 0, 0, N);
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			block_mean[j][i] = Block[j][i] - avg;
			block_mean[j][i] *= alpha;
			//block_mean[j][i] = MAX(0, block_mean[j][i]);
			//block_mean[j][i] = MIN(255, block_mean[j][i]);
		}
	}

}



void readBlock(int **image, int y, int x, int dy, int dx, int **block) {
	for (int j = 0; j < dy; j++) {
		for (int i = 0; i < dx; i++) {
			block[j][i] = image[y + j][x + i];
		}
	}
}

void writeBlock(int **image, int y, int x, int dy, int dx, int **block) {
	for (int j = 0; j < dy; j++) {
		for (int i = 0; i < dx; i++) {
			image[y + j][x + i] = block[j][i];
		}
	}
}
void Add_avg(int** Block, int N, int avg) {// 평균을 제거하는 함수-> block입력이 되면 평균 계산해서 원래 밝기에서 평균을 뺌->그게  block_mean

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			Block[j][i] += avg;
			/*Block[j][i] = MAX(0, Block[j][i]);
			Block[j][i] = MIN(255, Block[j][i]);*/
		}
	}

}

/*
readblock해서 block하나 맹금 사이즈 nxn
block[][]의 평균 계산; avg
block_mean[][]평균 제거 -1번

readblock-> 사이즈가 2n x2n ; Dblock
Dblock 평균 제거 -2번

1번과 2번의 차이 절댓값; error

한 픽셀 이동해서 반복-> 첫번째 블락에 대해 에러1/ 두번째 블락에 대해 에러1-> 에러가 제일 작은거 위치 찾음
=>1번 블락에 대한 최소 찾은겨

다음 1번블락과 안겹치게 2번블락으로 반복
->하나의 파란 블락에서 저장해야 하는거: 평균값, 위치(x,y) ->3개를 배열로 저장해놈
*/

int find_error(int** block1, int** block2, int height, int width) {
	int error = 0;
	for (int dy = 0; dy < height; dy++) {
		for (int dx = 0; dx < width; dx++) {
			error += abs(block1[dy][dx] - block2[dy][dx]);

		}
	}
	return error;
}


/*
scaling: Dblock2_mean- block_mean=error
Dblock2_mean->geometric_transform , scaling
scaling: Dblock2_mean*a -> a=0.3~0.4...1.0 -> 8개-> 최소가 되는 a도 저장


디코딩: 똑같은 사이즈의 다른 사진 가져와서
avg(block),x,y(Dblock) -> 가져와서  1/2만들어-> 평균을 제거한거에  ->a곱함->avg더함-> 첫번째에 놓음.

*/
double PSNR(int** im1, int** im2, int height, int width)
{
	double err = 0.0;
	for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
		err += ((double)im1[i][j] - im2[i][j]) * (im1[i][j] - im2[i][j]);
	}

	err = err / (width*height);

	return(10.0 * log10(255 * 255.0 / err));
}
void scaling(int** block, int** block_out, int N, float alpha) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			block_out[i][j] = block[i][j] * alpha;
		}
	}
}

void  findAlpha(int **Dblock_trans, int **Dblock2_scaling, int N, int **block_mean, ERROR* e, int* min, int dy, int dx, int ver) {
	int error = 0;
	for (float alpha = 0.3; alpha <= 1.1; alpha += 0.1) {
		scaling(Dblock_trans, Dblock2_scaling, N, alpha);
		error = find_error(block_mean, Dblock2_scaling, N, N);
		if (error < *min)
		{
			*min = error;
			e->x = dx;
			e->y = dy;
			e->trans_version = ver;
			e->a = alpha;
		}
	}
}

void findVersion(int **Dblock2_mean, int **Dblock_trans, int **Dblock2_scaling, int N, int **block_mean, ERROR* e, int* min, int dy, int dx) {
	for (int ver = 0; ver <= 7; ver++) {
		geometric_transform(Dblock2_mean, N, Dblock_trans, ver);
		findAlpha(Dblock_trans, Dblock2_scaling, N, block_mean, e, min, dy, dx, ver);
	}
}

void match_Dblock(int** image, int height, int width, int** Dblock, int **Dblock2, int **Dblock2_mean, int **Dblock_trans, int **Dblock2_scaling, int N, int **block_mean, ERROR* e, int* min) {
	for (int dy = 0; dy < height - 2 * N; dy ++ ) {
		for (int dx = 0; dx < width - 2 * N; dx ++ ) {
			readBlock(image, dy, dx, 2 * N, 2 * N, Dblock);
			downSize2(Dblock, Dblock2, 2, 2 * N, 2 * N);
			RemoveMean(Dblock2, N, Dblock2_mean, getBlockAvg(Dblock2,0,0,N)); 
			findVersion(Dblock2_mean, Dblock_trans, Dblock2_scaling, N, block_mean, e, min, dy, dx);
		}
	}
}
ERROR error_min[32][32]; // 32x32
typedef struct SORTED_ERROR_MIN {
	int ix;
	int iy;
	int error;
};

SORTED_ERROR_MIN sorted_error_min[1024];
void encoding(int** image, int height, int width, int N) {
	int **block = IntAlloc2(N, N);
	int **block_mean = IntAlloc2(N, N);
	int **Dblock = IntAlloc2(2 * N, 2 * N);
	int **Dblock_trans = IntAlloc2(N, N);
	int **Dblock2 = IntAlloc2(N, N);
	int **Dblock2_mean = IntAlloc2(N, N);
	int **Dblock2_scaling = IntAlloc2(N, N);
	int error = 0; ERROR e;

	int min;
	int ii = 0;
	int i = 0; int j = 0;
	int block_avg;
	for (int y = 0; y < height; y += N) {
		for (int x = 0; x < width; x += N) {
			readBlock(image, y, x, N, N, block);
			block_avg = prepared_Avg8x8[y][x];
			RemoveMean(block, N, block_mean, block_avg);
			min = INT_MAX;

			match_Dblock(image, height, width, Dblock, Dblock2, Dblock2_mean, Dblock_trans, Dblock2_scaling, N, block_mean, &e, &min);
			sorted_error_min[ii].error = min; sorted_error_min[ii].ix = x / N; sorted_error_min[ii++].iy = y / N;

			error_min[j][i].x = e.x;
			error_min[j][i].y = e.y;
			error_min[j][i].avg = getBlockAvg(block, 0, 0, N);
			error_min[j][i].trans_version = e.trans_version;
			error_min[j][i++].a = e.a;
			printf(" %d %d %d %d %d %f %d \n", y, x, e.x, e.y, e.trans_version, e.a, block_avg);
		}
		i = 0; j++;
	}

}

void decoding(int height, int width, int N, int** image_decoding, int** block_decoding, int **temp, int** block2_decoding, int** image_out) {
	for (int y = 0; y < height; y += N) {
		for (int x = 0; x < width; x += N) {
			readBlock(image_decoding, error_min[y / N][x / N].y, error_min[y / N][x / N].x, 2 * N, 2 * N, block_decoding);
			geometric_transform(block_decoding, 2 * N, temp, error_min[y / N][x / N].trans_version);
			downSize2(temp, block2_decoding, 2, 2 * N, 2 * N);
			RemoveMean_alpha(block2_decoding, N, block2_decoding, error_min[y / N][x / N].a);
			Add_avg(block2_decoding, N, error_min[y / N][x / N].avg);
			writeBlock(image_out, y, x, N, N, block2_decoding);
		}
	}
}

void sort_errorMin() {
	int max;
	SORTED_ERROR_MIN temp;
	int max_index = 0;

	for (int i = 0; i < 1024 - 1; i++) {
		max = sorted_error_min[i].error;
		for (int j = i + 1; j < 1024; j++) {
			if (max < sorted_error_min[j].error) {
				max_index = j;
				max = sorted_error_min[j].error;
			}
		}
		//swap i<->j
		temp.error = sorted_error_min[i].error;
		sorted_error_min[i].error = sorted_error_min[max_index].error;
		sorted_error_min[max_index].error = temp.error;

		temp.ix = sorted_error_min[i].ix;
		sorted_error_min[i].ix = sorted_error_min[max_index].ix;
		sorted_error_min[max_index].ix = temp.ix;

		temp.iy = sorted_error_min[i].iy;
		sorted_error_min[i].iy = sorted_error_min[max_index].iy;
		sorted_error_min[max_index].iy = temp.iy;
	}
}
ERROR error_min4x4[816];
void encoding4x4(int** image, int height, int width, int N) {
	int **block = IntAlloc2(N, N);
	int **block_mean = IntAlloc2(N, N);
	int **Dblock = IntAlloc2(2 * N, 2 * N);
	int **Dblock_trans = IntAlloc2(N, N);
	int **Dblock2 = IntAlloc2(N, N);
	int **Dblock2_mean = IntAlloc2(N, N);
	int **Dblock2_scaling = IntAlloc2(N, N);
	int error = 0; ERROR e;

	int min;
	int i = 0;
	int prepared_avg;
	int dy, dx; int count = 0;
	for (int i = 0; i < 204 * 4; i++)
	{
		switch (count)
		{
		case 0:
		{dy = 0; dx = 0; break; }
		case 1:
		{dy = 0; dx = 4; break; }
		case 2:
		{dy = 4; dx = 0; break; }
		case 3:
		{dy = 4; dx = 4; count = -1;  break; }
		} count++;

		readBlock(image, sorted_error_min[i / N].iy * 8 + dy, sorted_error_min[i / N].ix * 8 + dx, N, N, block);
		prepared_avg = prepared_Avg4x4[sorted_error_min[i / N].iy * 8 + dy][sorted_error_min[i / N].ix * 8 + dx];
		RemoveMean(block, N, block_mean, prepared_avg);
		min = INT_MAX;

		match_Dblock(image, height, width, Dblock, Dblock2, Dblock2_mean, Dblock_trans, Dblock2_scaling, N, block_mean, &e, &min);

		error_min4x4[i].x = e.x;
		error_min4x4[i].y = e.y;
		error_min4x4[i].avg = getBlockAvg(block, 0, 0, N);
		error_min4x4[i].trans_version = e.trans_version;
		error_min4x4[i].a = e.a;
		printf("--i:%d x : %d y : %d version :%d alpha : %.1f\n", i, e.x, e.y, e.trans_version, e.a);
	}


}

void decoding4x4(int height, int width, int N, int** image_decoding, int** block_decoding, int **temp, int** block2_decoding, int** image_out) {
	int dy, dx; int count = 0;
	for (int i = 0; i < 204 * 4; i++) {
		switch (count)
		{
		case 0:
		{dy = 0; dx = 0; break; }
		case 1:
		{dy = 0; dx = 4; break; }
		case 2:
		{dy = 4; dx = 0; break; }
		case 3:
		{dy = 4; dx = 4; count = -1;  break; }
		} count++;

		readBlock(image_decoding, error_min4x4[i].y, error_min4x4[i].x, 2 * N, 2 * N, block_decoding);
		geometric_transform(block_decoding, 2 * N, temp, error_min4x4[i].trans_version);
		downSize2(temp, block2_decoding, 2, 2 * N, 2 * N);
		RemoveMean_alpha(block2_decoding, N, block2_decoding, error_min4x4[i].a);
		Add_avg(block2_decoding, N, error_min4x4[i].avg);
		writeBlock(image_out, sorted_error_min[i / 4].iy * 8 + dy, sorted_error_min[i / 4].ix * 8 + dx, N, N, block2_decoding);
	}
}
void prepare_blockAvg(int** image, int** preparedAvg, int N, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			preparedAvg[y][x] = getBlockAvg(image, y, x, N);
		}
	}
}

int main() {
	clock_t start, end;
	start = clock();
	int height, width;
	int** image_decoding = ReadImage("images.jpg", &height, &width);
	int** image = ReadImage("LENA256.bmp", &height, &width);
	int** image_out = IntAlloc2(height, width);
	int N = 8;

	prepared_Avg16x16 = IntAlloc2(height, width);
	prepare_blockAvg(image, prepared_Avg16x16, 2 * N, height, width);
	prepared_Avg8x8 = IntAlloc2(height, width);
	prepare_blockAvg(image, prepared_Avg8x8, N, height, width);
	prepared_Avg4x4 = IntAlloc2(height, width);
	prepare_blockAvg(image, prepared_Avg4x4, N / 2, height, width);


	//인코딩 
	encoding(image, height, width, N);
	sort_errorMin();
	int ii = 0;
	ImageShow("본래 이미지", image, height, width);

	//디코딩
	int **block_decoding = IntAlloc2(2 * N, 2 * N);
	int **temp = IntAlloc2(2 * N, 2 * N);
	int **block2_decoding = IntAlloc2(2 * N, 2 * N);

	decoding(height, width, N, image_decoding, block_decoding, temp, block2_decoding, image_out);
	for (int i = 0; i < 3; i++) {
		decoding(height, width, N, image_out, block_decoding, temp, block2_decoding, image_out);
		printf("\n PSNR = %f", PSNR(image, image_out, height, width));
	}
	ImageShow("decoding", image_out, height, width);

	//4x4 인코딩
	encoding4x4(image, height, width, 4);
	//4x4 디코딩
	N = 4;
	block_decoding = IntAlloc2(2 * N, 2 * N);
	temp = IntAlloc2(2 * N, 2 * N);
	block2_decoding = IntAlloc2(2 * N, 2 * N);
	for (int i = 0; i < 3; i++) {
		decoding4x4(height, width, N, image_out, block_decoding, temp, block2_decoding, image_out);
		printf("\n PSNR = %f", PSNR(image, image_out, height, width));
	}
	ImageShow("20% decoding", image_out, height, width);

	end = clock();
	float res = (float)((end - start) / CLOCKS_PER_SEC);
	printf("실행시간 : %.3f", res);
}