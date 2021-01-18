#include <iostream>
#include <memory>
#include "cuda.h"
bool allclose(float x, float y, float threshold = 1e-4) {
	return abs(x - y) < threshold;
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
          int i = xzi * ywa + ywi;
          int xyi = xi * ya + yi;
          int zwi = zi * wa + wi;
          dst[i] = src1[xyi] * src2[zwi];
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
					int i = xzi * ywa + ywi;
					int xyi = xi * ya + yi;
					int zwi = zi * wa + wi;
	 				dst[i] = src1[xyi] * src2[zwi];
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
	const int defdim = 89;
	int xa = argc > 1 ? std::atoi(argv[1]) : defdim;
  int ya = argc > 2 ? std::atoi(argv[2]) : defdim;
	int za = argc > 3 ? std::atoi(argv[3]) : defdim;
  int wa = argc > 4 ? std::atoi(argv[4]) : defdim;
	float* src1 = (float*)malloc(sizeof(float) * xa * ya);
	float* src2 = (float*)malloc(sizeof(float) * za * wa);
	float* dst = (float*)malloc(sizeof(float)* xa * ya * za * wa);
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
	dotc(dst, src1 ,src2, xa, ya, za, wa);
	double cstime = checkpointc();
	std::cout<<"******CPU DST*********"<<std::endl;
	int xza = xa * za;
	int ywa = ya * wa;
	/*for(int xzi = 0; xzi < xza; xzi++) {
		for(int ywi = 0; ywi < ywa; ywi++) {
			std::cout<<dst[xzi * ywa + ywi]<<" ";
		}
		std::cout<<std::endl;
	}*/
	memset(dst, 0, sizeof(float)* xa * ya * za * wa);
	dim3 grids, blocks;
	grids.x = 1;
	grids.y = 1;
	grids.z = 1;
	blocks.x = 1;
	blocks.y = 1;
	blocks.z = 1;
	checkpointc();
	std::cout<<"SRC -> GPU ->DST"<<std::endl;
	dotg<<<grids, blocks>>>(dst, src1, src2, xa, ya, za, wa);
	double dstime = checkpointc();
	std::cout<<"CSTIME: "<<cstime<<std::endl;
	std::cout<<"DSTIME: "<<dstime<<std::endl;
}
