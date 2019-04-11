#include <stdio.h>
#include <assert.h>

#include "snde_types.h"
#include "geometry_types.h"
#include "vecops.h"

int main(int argc, char *argv[])
{
  snde_coord A2x2[4] = { .614, .739, .767, .95 }; // NOTE: stored column major (fortran order)
  snde_coord b2[2] = { 1.0, -1.0};
  size_t pivots2[2];

  fmatrixsolve(A2x2,b2,2,1,pivots2);
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
  
  fmatrixsolve(A4x4,b4,4,4,pivots4);
  printf("pivots=%d %d %d %d\n",(int)pivots4[0],(int)pivots4[1],(int)pivots4[2],(int)pivots4[3]);
  for (int col=0;col < 4; col++) {
    printf("%10f %10f %10f %10f\n",(float)b4[0 + col*4],(float)b4[1 + col*4],(float)b4[2 + col*4],(float)b4[3 + col*4]);
  }

  assert(fabs(b4[0]-1.0907e1) < .001);
  assert(fabs(b4[3]-1.5926) < .001);
  assert(fabs(b4[4]+2.4815e1) < .001);
  assert(fabs(b4[15]+1.00) < .001);

  
  return 0;
}
