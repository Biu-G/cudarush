#include <iostream>
#include <cstdio>
#include <memory>
#include "cuda.h"
#define PR
bool fcc(float x, float y, float threshold = 1e-4) {
  return abs(x - y) < threshold;
}

bool allclose(float*x, float*y, int s) {
	for(int i=0;i<s;i++){
		float fx = x[i];
		float fy = y[i];
		bool fc = fcc(fx, fy);
		if(!fc) {
			return false;
		}
	}
	return true;
}

__global__ void dotg(float*dst, float*src1, float*src2, int xa, int ya, int za, int wa) {
  //xy * zw -> xz * yw
  int ywa = ya * wa;
  for(int xi = 0; xi < xa; xi++) {
    for(int yi = 0; yi < ya; yi++) {
      for(int zi = 0; zi < za; zi++) {
        for(int wi = 0; wi < wa; wi++) {
          int xzi = xi * za + zi;
          int ywi = yi * wa + wi;
          int ii = xzi * ywa + ywi;
          int xyi = xi * ya + yi;
          int zwi = zi * wa + wi;
          dst[ii] = src1[xyi] * src2[zwi];
        }
      }
    }
  }
}

void dotc(float*dst, float*src1, float*src2, int xa, int ya, int za, int wa) {
	//xy * zw -> xz * yw
	int ywa = ya * wa;
	for(int xi = 0; xi < xa; xi++) {
		for(int yi = 0; yi < ya; yi++) {
			for(int zi = 0; zi < za; zi++) {
				for(int wi = 0; wi < wa; wi++) {
					int xzi = xi * za + zi;
					int ywi = yi * wa + wi;
					int ii = xzi * ywa + ywi;
					int xyi = xi * ya + yi;
					int zwi = zi * wa + wi;
	 				dst[ii] = src1[xyi] * src2[zwi];
	 			}
			}
		}
	}
}

double checkpointc(int us = 1e3) {
	static clock_t timer = -1;
	clock_t newtime = clock();
	double ustime = us * ((newtime - timer) / (double)CLOCKS_PER_SEC);
	timer = newtime;
	return ustime;
}
	

int main(int argc, char* argv[]) {
	const int defdim = 39;
	int xa = argc > 1 ? std::atoi(argv[1]) : defdim;
  int ya = argc > 2 ? std::atoi(argv[2]) : defdim;
	int za = argc > 3 ? std::atoi(argv[3]) : defdim;
  int wa = argc > 4 ? std::atoi(argv[4]) : defdim;
	int aa = xa * ya * za * wa;
	float* src1 = (float*)malloc(sizeof(float) * xa * ya);
	float* src2 = (float*)malloc(sizeof(float) * za * wa);
	float* dstc = (float*)malloc(sizeof(float)* xa * ya * za * wa);
	float* dsrc1, *dsrc2, *dstd;
	cudaMalloc((void**)&dsrc1, sizeof(float) * xa * ya);
  cudaMalloc((void**)&dsrc2, sizeof(float) * za * wa);
	cudaMalloc((void**)&dstd, sizeof(float) * aa);
  /*float* dsrc1 = (float*)malloc(sizeof(float) * xa * ya);
  float* dsrc2 = (float*)malloc(sizeof(float) * za * wa);
	float* dstd = (float*)malloc(sizeof(float)* xa * ya * za * wa);*/
	float* dstc2 = (float*)malloc(sizeof(float)* aa);
	std::cout<<"------SRC1 / SRC2--------"<<std::endl;
	for(int xi = 0; xi < xa; xi++) {
		for(int yi = 0; yi < ya; yi++) {
			src1[xi * ya + yi] = xi * ya + yi;
			std::cout<<src1[xi * ya + yi]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<"+++++THRES+++++++"<<std::endl;
	  std::cout<<"------SRC1 / SRC2--------"<<std::endl;
  for(int zi = 0; zi < za; zi++) {
    for(int wi = 0; wi < wa; wi++) {
			src2[zi * wa + wi] = zi * wa + wi;
      std::cout<<src2[zi * wa + wi]<<" ";
    }
		std::cout<<std::endl;
	}
	std::cout<<"SRC ->CPU -> DST"<<std::endl;
	checkpointc();
	dotc(dstc, src1 ,src2, xa, ya, za, wa);
	double cstime = checkpointc();
	std::cout<<"******CPU DST*********"<<std::endl;
	int xza = xa * za;
	int ywa = ya * wa;
#if 0
	for(int xzi = 0; xzi < xza; xzi++) {
		for(int ywi = 0; ywi < ywa; ywi++) {
			std::cout<<dstc[xzi * ywa + ywi]<<" ";
		}
		std::cout<<std::endl;
	}
#endif
	dim3 grids, blocks;
	grids.x = 1;
	grids.y = 1;
	grids.z = 1;
	blocks.x = 1;
	blocks.y = 1;
	blocks.z = 1;
	std::cout<<"SRC -> GPU ->DST"<<std::endl;
	checkpointc();
	cudaMemcpy(dsrc1, src1, sizeof(float) * xa * ya, cudaMemcpyHostToDevice);
	cudaMemcpy(dsrc2, src2, sizeof(float) * za * wa, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	clock_t pretime = clock();
	dotg<<<grids, blocks>>>(dstd, dsrc1, dsrc2, xa, ya, za, wa);
	cudaDeviceSynchronize();
	clock_t aftime = clock();
	double coretime = 1e3 * ((aftime - pretime) / (double)CLOCKS_PER_SEC);
	cudaMemcpy(dstc2, dstd, sizeof(float) * aa, cudaMemcpyDeviceToHost);
	double dstime = checkpointc();
	std::cout<<"******GPU DST*******"<<std::endl;
#if 0
  for(int xzi = 0; xzi < xza; xzi++) {
    for(int ywi = 0; ywi < ywa; ywi++) {
      std::cout<<dstc2[xzi * ywa + ywi]<<" ";
    }
    std::cout<<std::endl;
  }
#endif
	printf("DOTMUL: %d^%d * %d^%d\n", xa, ya, za, wa);
	std::cout<<"CSTIME: "<<cstime<<std::endl;
	std::cout<<"CPY-DET: "<<dstime - coretime<<std::endl;
	std::cout<<"CORETIME: "<<coretime<<std::endl;
	bool alc = allclose(dstc, dstc2, aa);
	std::cout<<"ALC: "<<alc<<std::endl;
}
