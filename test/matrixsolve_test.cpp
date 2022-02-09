#include <stdio.h>
#include <assert.h>

#include "snde_types.h"
#include "geometry_types.h"
#include "vecops.h"

int main(int argc, char *argv[])
{
  snde_coord A2x2[4] = { .614f, .739f, .767f, .95f }; // NOTE: stored column major (fortran order)
  snde_coord b2[2] = { 1.0f, -1.0f};
  size_t pivots2[2];

  fmatrixsolve(A2x2,b2,2,1,pivots2,false);
  // MATLAB/Octave verification code:
  // A2x2= [.614, .739 ; .767, .95]';
  // b2 = [1.0; -1.0];
  // A2x2 \ b2  %  gives 104.143, -82.065

  printf("x = { %f, %f } (compare with { 104.143, -82.065 })\n",(float)b2[0],(float)b2[1]);
  
  assert(fabs(b2[0]-104.143) < .01);
  assert(fabs(b2[1]+82.065) < .01);
  

  snde_coord A4x4[16] = { 1, 2, 3, 4,
			  5, 7, 9, 12,
			  9, -10, -11, 14,
			  13,14,15,19}; // NOTE: stored column major (fortran order)
  snde_coord b4[16] = { 1.0, 0.0, 0.0,0.0,
			0.0, 1.0, 0.0,0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0};
  size_t pivots4[4];
  // MATLAB/Octave verification code: inv([ 1 2 3 4 ; 5 7 9 12; 9 -10 -11 14 ; 13 14 15 19]')'
  
  fmatrixsolve(A4x4,b4,4,4,pivots4,false);
  printf("pivots=%d %d %d %d\n",(int)pivots4[0],(int)pivots4[1],(int)pivots4[2],(int)pivots4[3]);
  for (int col=0;col < 4; col++) {
    printf("%10f %10f %10f %10f\n",(float)b4[0 + col*4],(float)b4[1 + col*4],(float)b4[2 + col*4],(float)b4[3 + col*4]);
  }
  
  assert(fabs(b4[0]-1.0907e1) < .002);
  assert(fabs(b4[3]-1.5926) < .002);
  assert(fabs(b4[4]+2.4815e1) < .002);
  assert(fabs(b4[15]+1.00) < .002);
  


  //240351

  
  snde_coord A6x6[36] = {
    0.0015553981f,        0, -0.0040931798f,        0, 0.0025377816f,        0,
    0.0068715753f,        0, -0.0010180307f,        0, -0.0058535446f,        0,
    1,        0,        1,        0,        1,        0,
    0, 0.0015553981f,        0, -0.0040931798f,        0, 0.0025377816f,
    0, 0.0068715753f,        0, -0.0010180307f,        0, -0.0058535446f,
    0,        1,        0,        1,        0,        1	
  };// NOTE: stored column major (fortran order)
  //snde_coord b6[36] = {
  //  1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  //  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  //  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  //  0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
  //  0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  //  0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
  //};
    snde_coord b6[6]={ 23.3592f, -29.5544f,  34.1624f,  -29.452f,  34.1624f, -20.3896f}; 
  size_t pivots6[6];
  // MATLAB/Octave verification code: inv([ 1 2 3 4 ; 5 7 9 12; 9 -10 -11 14 ; 13 14 15 19]')'
  
  fmatrixsolve(A6x6,b6,6,/*6*/1,pivots6,false);
  printf("pivots=%d %d %d %d %d %d\n",(int)pivots6[0],(int)pivots6[1],(int)pivots6[2],(int)pivots6[3],(int)pivots6[4],(int)pivots6[5]);
  for (int col=0;col < 6; col++) {
    printf("%10f %10f %10f %10f %10f %10f\n",(float)b6[0 + col*6],(float)b6[1 + col*6],(float)b6[2 + col*6],(float)b6[3 + col*6],(float)b6[4 + col*6],(float)b6[5 + col*6]);
  }

  assert(fabs(b6[0]+656.026) < .002);
  assert(fabs(b6[1]+899.612) < .002);
  assert(fabs(b6[2]-30.561) < .002);
  assert(fabs(b6[3]-891.675) < .002);
  assert(fabs(b6[4]+651.376) < .002);
  assert(fabs(b6[5]+26.465) < .002);

  
  
  return 0;
}
