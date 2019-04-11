#ifndef __OPENCL_VERSION__
#include <errno.h>
#include <stddef.h>
#include <math.h>
#endif

#ifndef SNDE_VECOPS_H
#define SNDE_VECOPS_H

#ifdef _MSC_VER
#define VECOPS_INLINE  __inline
#else
#define VECOPS_INLINE  inline
#endif

#ifdef __OPENCL_VERSION__
/* if this is an opencl kernel */

#ifdef __ENDIAN_LITTLE__
#define MY_INFNAN_LITTLE_ENDIAN
#endif 

#define ERANGE 34

typedef __constant unsigned char my_infnan_constchar_t;
typedef __constant float *my_infnan_float32_ptr_t;

#else

#if !(defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN) && !defined(__BIG_ENDIAN__)
#define MY_INFNAN_LITTLE_ENDIAN
#endif


typedef uint8_t my_infnan_constchar_t;
typedef float32_t *my_infnan_float32_ptr_t;

#endif /* __OPENCL_VERSION__ */

#ifdef MY_INFNAN_LITTLE_ENDIAN
static const my_infnan_constchar_t NaNconst[4]={ 0x00, 0x00, 0xc0, 0x7f };
static const my_infnan_constchar_t Infconst[4]={ 0x00, 0x00, 0x80, 0x7f };
static const my_infnan_constchar_t NegInfconst[4]={ 0x00, 0x00, 0x80, 0xff };
#else
static const my_infnan_constchar_t NaNconst[4]={ 0x7f,0xc0,0x00,0x00 };
static const my_infnan_constchar_t Infconst[4]={ 0x7f,0x80,0x00,0x00 };
static const my_infnan_constchar_t NegInfconst[4]={ 0xff,0x80,0x00,0x00 };
#endif

static VECOPS_INLINE snde_coord my_infnan(int error) /* be sure to disable SIGFPE */
{
  
  if (error==ERANGE) return *((my_infnan_float32_ptr_t)&Infconst);
  else if (error==-ERANGE) return *((my_infnan_float32_ptr_t)&NegInfconst);
  else return *((my_infnan_float32_ptr_t)&NaNconst);
}



static VECOPS_INLINE void multcmat23coord(snde_cmat23 cmat,snde_coord3 vec,snde_coord2 *out)
/* Multiply 2x3 matrix by 3-coord, giving 2-vector */
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out->coord[outel]=0.0;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out->coord[outel] = out->coord[outel] + cmat.row[outel].coord[sumidx]*vec.coord[sumidx];
    }
  }
}

static VECOPS_INLINE void multcmat23transposecoord(snde_cmat23 cmat,snde_coord2 vec,snde_coord3 *out)
/* Multiply transpose of 2x3 matrix by 2-coord, giving 3-vector */
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 3; outel++) {
    out->coord[outel]=0.0;
    for (sumidx=0;sumidx < 2; sumidx++) {
      out->coord[outel] = out->coord[outel] + cmat.row[sumidx].coord[outel]*vec.coord[sumidx];
    }
  }
}

static VECOPS_INLINE void multcmat23vec(snde_coord *mat,snde_coord *vec,snde_coord *out)
/* Multiply 2x3 matrix by 3-vector, giving 2-vector */
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out[outel] = out[outel] + mat[ outel*3 + sumidx]*vec[sumidx];
    }
  }
}


static VECOPS_INLINE void multveccmat23(snde_coord *vec,snde_coord *mat,snde_coord *out)
/* Multiply 2-vector by 2x3 matrix giving 3-vector  */
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 3; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 2; sumidx++) {
      out[outel] = out[outel] + mat[ sumidx*3 + outel]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multcmatvec4(snde_coord *mat,snde_coord *vec,snde_coord *out)
// mat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 4; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 4; sumidx++) {
      out[outel] = out[outel] + mat[ outel*4 + sumidx]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multcmatvec3(snde_coord *mat,snde_coord *vec,snde_coord *out)
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 3; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out[outel] = out[outel] + mat[ outel*3 + sumidx]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multmatvec2(snde_coord *mat,snde_coord *vec,snde_coord *out)
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 2; sumidx++) {
      out[outel] = out[outel] + mat[ outel*2 + sumidx]*vec[sumidx];
    }
  }
}


static VECOPS_INLINE snde_coord dotvecvec3(snde_coord *vec1,snde_coord *vec2)
{
  int sumidx;
  snde_coord val=0.0;
  for (sumidx=0;sumidx < 3; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
  }
  return val;
}

static VECOPS_INLINE snde_coord dotcoordcoord3(snde_coord3 vec1,snde_coord3 vec2)
{
  int sumidx;
  snde_coord val=0.0;
  for (sumidx=0;sumidx < 3; sumidx++) {
    val = val + vec1.coord[sumidx]*vec2.coord[sumidx];
    
  }
  return val;
}


static VECOPS_INLINE snde_coord dotvecvec2(snde_coord *vec1,snde_coord *vec2)
{
  int sumidx;
  snde_coord val=0.0;
  for (sumidx=0;sumidx < 2; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
  }
  return val;
}


static VECOPS_INLINE snde_coord dotcoordcoord2(snde_coord2 vec1,snde_coord2 vec2)
{
  int sumidx;
  snde_coord val=0.0;
  for (sumidx=0;sumidx < 2; sumidx++) {
    val = val + vec1.coord[sumidx]*vec2.coord[sumidx];
    
  }
  return val;
}


static VECOPS_INLINE void scalevec3(snde_coord coeff,snde_coord *vec1,snde_coord *out)
{
  size_t cnt;
  for (cnt=0;cnt < 3; cnt++) {
    out[cnt]=coeff*vec1[cnt];
  }
}

static VECOPS_INLINE void scalecoord3(snde_coord coeff,snde_coord3 vec1,snde_coord3 *out)
{
  size_t cnt;
  for (cnt=0;cnt < 3; cnt++) {
    out->coord[cnt]=coeff*vec1.coord[cnt];
  }
}


static VECOPS_INLINE void scalevec2(snde_coord coeff,snde_coord *vec1,snde_coord *out)
{
  size_t cnt;
  for (cnt=0;cnt < 2; cnt++) {
    out[cnt]=coeff*vec1[cnt];
  }
}

static VECOPS_INLINE void scalecoord2(snde_coord coeff,snde_coord2 vec1,snde_coord2 *out)
{
  size_t cnt;
  for (cnt=0;cnt < 2; cnt++) {
    out->coord[cnt]=coeff*vec1.coord[cnt];
  }
}



static VECOPS_INLINE void subvecvec3(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void subcoordcoord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] - vec2.coord[outidx];
    
  }
}

static VECOPS_INLINE void subvecvec2(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 2; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void subcoordcoord2(snde_coord2 vec1,snde_coord2 vec2,snde_coord2 *out)
{
  int outidx;

  for (outidx=0;outidx < 2; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] - vec2.coord[outidx];
    
  }
}

static VECOPS_INLINE void addvecscaledvec3(snde_coord *vec1,snde_coord coeff, snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] + coeff* vec2[outidx];
    
  }
}

static VECOPS_INLINE void addcoordscaledcoord3(snde_coord3 vec1,snde_coord coeff, snde_coord3 vec2,snde_coord3 *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] + coeff* vec2.coord[outidx];
    
  }
}



static VECOPS_INLINE void normalize_wcoord4(snde_coord *vec)
/* operates in-place */
{
  vec[0] /= vec[3];
  vec[1] /= vec[3];
  vec[2] /= vec[3];
  vec[3] = 1.0;
  
}



static VECOPS_INLINE snde_coord to_unit_vector4(snde_coord *vec)
/* operates in-place... returns scaling factor */
{
  snde_coord factor;

  factor=1.0/sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
#ifdef __OPENCL_VERSION__
  /* if this is an opencl kernel, a W component makes the result invalid */
  if (vec[3] != 0.0) {
    factor = my_infnan(0); // NaN factor
  }
#else
  assert(vec[3]==0.0); /* vectors should have no 'w' component */
#endif
  vec[0] *= factor;
  vec[1] *= factor;
  vec[2] *= factor;
  //vec[3] *= factor;

  return factor;
  
}


static VECOPS_INLINE snde_coord normvec3(snde_coord *vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  return factor;
}

static VECOPS_INLINE snde_coord normcoord3(snde_coord3 vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=sqrt(vec.coord[0]*vec.coord[0]+vec.coord[1]*vec.coord[1]+vec.coord[2]*vec.coord[2]);
  return factor;
}

static VECOPS_INLINE void normalizevec3(snde_coord *vec)
/* in-place vector normalization */
{
  snde_coord factor;

  factor=normvec3(vec);
  vec[0] /= factor;
  vec[1] /= factor;
  vec[2] /= factor;
}

static VECOPS_INLINE void normalizecoord3(snde_coord3 *vec)
/* in-place vector normalization */
{
  snde_coord factor;

  factor=normcoord3(*vec);
  vec->coord[0] /= factor;
  vec->coord[1] /= factor;
  vec->coord[2] /= factor;
}


static VECOPS_INLINE snde_coord normvec2(snde_coord *vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=sqrt(vec[0]*vec[0]+vec[1]*vec[1]);
  return factor;
}

static VECOPS_INLINE snde_coord normcoord2(snde_coord2 vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=sqrt(vec.coord[0]*vec.coord[0]+vec.coord[1]*vec.coord[1]);
  return factor;
}

static VECOPS_INLINE void normalizevec2(snde_coord *vec)
/* in-place vector normalization */
{
  snde_coord factor;

  factor=normvec2(vec);
  vec[0] /= factor;
  vec[1] /= factor;
}

static VECOPS_INLINE void normalizecoord2(snde_coord2 *vec)
/* in-place vector normalization */
{
  snde_coord factor;

  factor=normcoord2(*vec);
  vec->coord[0] /= factor;
  vec->coord[1] /= factor;
}


static VECOPS_INLINE snde_coord to_unit_vector3(snde_coord *vec)
/* operates in-place */
{
  snde_coord factor;

  factor=1.0/sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  vec[0] *= factor;
  vec[1] *= factor;
  vec[2] *= factor;

  return factor;
  
}

static VECOPS_INLINE snde_coord to_unit_coord3(snde_coord3 *vec)
/* operates in-place */
{
  snde_coord factor;

  factor=1.0/sqrt(vec->coord[0]*vec->coord[0]+vec->coord[1]*vec->coord[1]+vec->coord[2]*vec->coord[2]);
  vec->coord[0] *= factor;
  vec->coord[1] *= factor;
  vec->coord[2] *= factor;

  return factor;
  
}


static VECOPS_INLINE void sign_nonzero3(snde_coord *input,snde_coord *output)
{
  int cnt;
  for (cnt=0;cnt < 3;cnt++) {
    if (input[cnt] < 0.0) output[cnt]=-1.0;
    else output[cnt]=1.0;
  }
}

static VECOPS_INLINE void sign_nonzerocoord3(snde_coord3 input,snde_coord3 *output)
{
  int cnt;
  for (cnt=0;cnt < 3;cnt++) {
    if (input.coord[cnt] < 0.0) output->coord[cnt]=-1.0;
    else output->coord[cnt]=1.0;
  }
}


static VECOPS_INLINE void multvecvec3(snde_coord *vec1,snde_coord *vec2,snde_coord *output)
{
  int cnt;
  for (cnt=0;cnt < 3; cnt++) {
    output[cnt]=vec1[cnt]*vec2[cnt];
  }
}

static VECOPS_INLINE void multcoordcoord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *output)
{
  int cnt;
  for (cnt=0;cnt < 3; cnt++) {
    output->coord[cnt]=vec1.coord[cnt]*vec2.coord[cnt];
  }
}

static VECOPS_INLINE void crossvecvec3(snde_coord *vec1,snde_coord *vec2,snde_coord *output)
{
  /* 
     vec1 cross vec2 

   |   i     j     k    |
   | vec10 vec11  vec12 |
   | vec20 vec21  vec22 |
  */
  output[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
  output[1] = vec1[2]*vec2[0]-vec1[0]*vec2[2];
  output[2] = vec1[0]*vec2[1]-vec1[1]*vec2[0];
}

static VECOPS_INLINE void crosscoordcoord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *output)
{
  /* 
     vec1 cross vec2 

   |   i     j     k    |
   | vec10 vec11  vec12 |
   | vec20 vec21  vec22 |
  */
  output->coord[0] = vec1.coord[1]*vec2.coord[2]-vec1.coord[2]*vec2.coord[1];
  output->coord[1] = vec1.coord[2]*vec2.coord[0]-vec1.coord[0]*vec2.coord[2];
  output->coord[2] = vec1.coord[0]*vec2.coord[1]-vec1.coord[1]*vec2.coord[0];
}

static VECOPS_INLINE snde_coord crossvecvec2(snde_coord *vec1,snde_coord *vec2)
{
  /* 
     vec1 cross vec2 

   |   i     j     k    |
   | vec10 vec11        |
   | vec20 vec21        |
  */
  return vec1[0]*vec2[1]-vec1[1]*vec2[0];
}

static VECOPS_INLINE snde_coord crosscoordcoord2(snde_coord2 vec1,snde_coord2 vec2)
{
  /* 
     vec1 cross vec2 

   |   i     j     k    |
   | vec10 vec11        |
   | vec20 vec21        |
  */
  return vec1.coord[0]*vec2.coord[1]-vec1.coord[1]*vec2.coord[0];
}


static VECOPS_INLINE void mean2vec3(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int cnt;
  
  for (cnt=0;cnt < 3; cnt++) {
    out[cnt]=(vec1[cnt]+vec2[cnt])/2.0;
  }
}

static VECOPS_INLINE void mean2coord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *out)
{
  int cnt;
  
  for (cnt=0;cnt < 3; cnt++) {
    out->coord[cnt]=(vec1.coord[cnt]+vec2.coord[cnt])/2.0;
  }
}

static VECOPS_INLINE void fmatrixsolve(snde_coord *A,snde_coord *b,size_t n,size_t nsolve,size_t *pivots)
// solves A*x=b, where A is n*n, b is n*nsolve, and x is n*1
// must provide a n-length vector of size_t "pivots" that this routine uses for intermediate storage.
// *** NOTE: *** This routine will overwrite the contents of A and b... stores the
// result in b. 
// NOTE: A and b should be stored column major (Fortran style)
{
  size_t row,rsrch,col,succ_row,pred_row,rowcnt;
  snde_coord bestpivot,leading_val;
  size_t old_pivots_row;
  size_t solvecnt;
  snde_coord first_el,pred_val;
  int swapped;
  size_t swappedentry;
  
  // initialize blank pivots
  for (row=0; row < n; row++) {
    pivots[row]=row;
  }

  for (row=0; row < n; row++) {
    // find largest magnitude row
    old_pivots_row=pivots[row];
    bestpivot=fabs(A[pivots[row] + row*n]);  // pull out this diagonal etnry to start
    swapped=FALSE;
    for (rsrch=row+1; rsrch < n; rsrch++) {
      if (fabs(A[pivots[rsrch] + row*n]) > bestpivot) {
	bestpivot=fabs(A[pivots[rsrch] + row*n]);
	pivots[row]=pivots[rsrch];
	swappedentry=rsrch;
	swapped=TRUE;
      }
    }
    if (swapped) {
      pivots[swappedentry]=old_pivots_row; // complete swap
    }
    // Divide this row by its first element
    first_el = A[pivots[row] + row*n];
    A[pivots[row] + row*n]=1.0f;
    for (col=row+1;col < n; col++) {
      A[pivots[row] + col*n] /= first_el; 
    }
    for (solvecnt=0; solvecnt < nsolve; solvecnt++) {
      b[pivots[row]  + solvecnt*n] /= first_el;
    }
    
    // subtract a multiple of this row from all succeeding rows
    for (succ_row = row+1; succ_row < n; succ_row++) {
      leading_val = A[pivots[succ_row] + row*n];
      A[pivots[succ_row] + row*n]=0.0f;
      for (col=row+1; col < n; col++) {
	A[pivots[succ_row] + col*n] -= leading_val*A[pivots[row] + col*n];
      }
      for (solvecnt=0; solvecnt < nsolve; solvecnt++) {
	b[pivots[succ_row] + solvecnt*n] -= leading_val*b[pivots[row] + solvecnt*n];
      }
    }
  }

  // OK; now A should be upper-triangular
  // Now iterate through the back-substitution. 
  for (rowcnt=0; rowcnt < n; rowcnt++) { // 
    row=n-1-rowcnt;

    // subtract a multiple of this row
    // from all preceding rows

    for (pred_row=0; pred_row < row; pred_row++) {
      pred_val = A[pivots[pred_row] + row*n];
      A[pivots[pred_row] + row*n]=0.0f;

      // this loop is unnecessary because the row must be zero in the remaining columns
      //for (col=row+1; col < n; col++) {
      //  A[pivots[pred_row] + col*n] -= pred_val * A[pivots[row] + col*n];
      //}
      for (solvecnt=0; solvecnt < nsolve; solvecnt++) {
	b[pivots[pred_row] + solvecnt*n] -= pred_val * b[pivots[row] + solvecnt*n];
      }
    }
  }

  // ... solved! A should be the identity matrix and Answer should be stored in b...
  // But we need to reorder the rows to undo the pivot

  // go through each column of the answer,
  // moving it to the first column of A,
  // then copying it back in the correct order
  for (col=0; col < nsolve; col++) {
    for (row=0; row < n; row++) {
      A[row]=b[row + col*n];
    }

    for (row=0; row < n; row++) {
      b[row + col*n]=A[pivots[row]];
    }
    
  }
  
}


#endif // SNDE_VECOPS_H
