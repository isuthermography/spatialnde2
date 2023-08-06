#ifndef SNDE_VECOPS_H
#define SNDE_VECOPS_H

#ifndef __OPENCL_VERSION__
#include <errno.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

#include "snde/snde_types.h"
#endif


#ifdef _MSC_VER
#define VECOPS_INLINE  __inline
#else
#define VECOPS_INLINE  inline
#endif

#ifdef __OPENCL_VERSION__
/* if this is an opencl kernel */

#define VECOPS_GLOBAL __global
#define VECOPS_LOCAL __local

#ifdef __ENDIAN_LITTLE__
#define SNDE_INFNAN_LITTLE_ENDIAN
#endif 

#define ERANGE 34

typedef __constant unsigned char snde_infnan_constchar_t;
typedef __constant float *snde_infnan_float32_ptr_t;

#else

#define VECOPS_GLOBAL // no meaning in regular c
#define VECOPS_LOCAL // no meaning in regular c

#define VECOPS_DOUBLEPREC 1 // always include double precision functions on a real CPU

#if !(defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN) && !defined(__BIG_ENDIAN__)
#define SNDE_INFNAN_LITTLE_ENDIAN
#endif


typedef uint8_t snde_infnan_constchar_t;
typedef snde_float32 *snde_infnan_float32_ptr_t;

#endif /* __OPENCL_VERSION__ */

#ifdef SNDE_INFNAN_LITTLE_ENDIAN
static const snde_infnan_constchar_t NaNconst[4]={ 0x00, 0x00, 0xc0, 0x7f };
static const snde_infnan_constchar_t Infconst[4]={ 0x00, 0x00, 0x80, 0x7f };
static const snde_infnan_constchar_t NegInfconst[4]={ 0x00, 0x00, 0x80, 0xff };
#else
static const snde_infnan_constchar_t NaNconst[4]={ 0x7f,0xc0,0x00,0x00 };
static const snde_infnan_constchar_t Infconst[4]={ 0x7f,0x80,0x00,0x00 };
static const snde_infnan_constchar_t NegInfconst[4]={ 0xff,0x80,0x00,0x00 };
#endif

static VECOPS_INLINE snde_coord snde_infnan(int error) /* be sure to disable SIGFPE */
{
  
  if (error==ERANGE) return *((snde_infnan_float32_ptr_t)&Infconst);
  else if (error==-ERANGE) return *((snde_infnan_float32_ptr_t)&NegInfconst);
  else return *((snde_infnan_float32_ptr_t)&NaNconst);
}



static VECOPS_INLINE void multcmat23coord(snde_cmat23 cmat,snde_coord3 vec,snde_coord2 *out)
/* Multiply 2x3 matrix by 3-coord, giving 2-vector */
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out->coord[outel]=0.0f;
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
    out->coord[outel]=0.0f;
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
    out[outel]=0.0f;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out[outel] = out[outel] + mat[ outel*3 + sumidx]*vec[sumidx];
    }
  }
}


//static VECOPS_INLINE void multveccmat23(snde_coord *vec,snde_coord *mat,snde_coord *out)
/* Multiply 2-vector by 2x3 matrix giving 3-vector  */
// cmat stored row-major (C-style)
//{
//  int outel,sumidx;
//
//  for (outel=0;outel < 3; outel++) {
//    out[outel]=0.0;
//    for (sumidx=0;sumidx < 2; sumidx++) {
//      out[outel] = out[outel] + mat[ sumidx*3 + outel]*vec[sumidx];
//    }
//  }
//}

static VECOPS_INLINE void multcmatvec4(snde_coord *mat,snde_coord *vec,snde_coord *out)
// mat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 4; outel++) {
    out[outel]=0.0f;
    for (sumidx=0;sumidx < 4; sumidx++) {
      out[outel] = out[outel] + mat[ outel*4 + sumidx]*vec[sumidx];
    }
  }
}


static VECOPS_INLINE void multcmat44(snde_coord *mat1,snde_coord *mat2,snde_coord *out)
// mat1, mat2, out stored row-major (C-style)
{
  int outrow,outcol,sumidx;

  for (outcol=0;outcol < 4; outcol++) {
    for (outrow=0;outrow < 4; outrow++) {
      out[outrow*4 + outcol]=0.0f;
      for (sumidx=0;sumidx < 4; sumidx++) {
	out[outrow*4 + outcol] += mat1[ outrow*4 + sumidx]*mat2[sumidx*4 + outcol];
      }
    }
  }
}



static VECOPS_INLINE void multcmatvec3(snde_coord *mat,snde_coord *vec,snde_coord *out)
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 3; outel++) {
    out[outel]=0.0f;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out[outel] = out[outel] + mat[ outel*3 + sumidx]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multcmatvec2(snde_coord *mat,snde_coord *vec,snde_coord *out)
// cmat stored row-major (C-style)
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out[outel]=0.0f;
    for (sumidx=0;sumidx < 2; sumidx++) {
      out[outel] = out[outel] + mat[ outel*2 + sumidx]*vec[sumidx];
    }
  }
}


static VECOPS_INLINE void copyvecn(snde_coord *in, snde_coord *out, snde_index n)
{
  for (snde_index posn=0; posn < n; posn++) {
    out[posn]=in[posn];
  }
}

static VECOPS_INLINE void copyvecnglobal(snde_coord *in, OCL_GLOBAL_ADDR snde_coord *out, snde_index n)
{
  for (snde_index posn=0; posn < n; posn++) {
    out[posn]=in[posn];
  }
}

static VECOPS_INLINE snde_coord dotvecvec3(snde_coord *vec1,snde_coord *vec2)
{
  int sumidx;
  snde_coord val=0.0f;
  for (sumidx=0;sumidx < 3; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
  }
  return val;
}

static VECOPS_INLINE snde_coord dotcoordcoord3(snde_coord3 vec1,snde_coord3 vec2)
{
  int sumidx;
  snde_coord val=0.0f;
  for (sumidx=0;sumidx < 3; sumidx++) {
    val = val + vec1.coord[sumidx]*vec2.coord[sumidx];
    
  }
  return val;
}


static VECOPS_INLINE snde_coord dotvecvec2(snde_coord *vec1,snde_coord *vec2)
{
  int sumidx;
  snde_coord val=0.0f;
  for (sumidx=0;sumidx < 2; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
  }
  return val;
}


static VECOPS_INLINE snde_coord dotcoordcoord2(snde_coord2 vec1,snde_coord2 vec2)
{
  int sumidx;
  snde_coord val=0.0f;
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
// For 3D non-projective coordinates only!!!
{
  size_t cnt;
  for (cnt=0;cnt < 3; cnt++) {
    out->coord[cnt]=coeff*vec1.coord[cnt];
  }
}

static VECOPS_INLINE void scalecoord4(snde_coord coeff,snde_coord4 vec1,snde_coord4 *out)
// if vec1 is 3D written in 4D projective space, it must be a vector not a position!
{
  size_t cnt;
  for (cnt=0;cnt < 4; cnt++) {
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



static VECOPS_INLINE snde_coord distsqglobalvecn(VECOPS_GLOBAL snde_coord *vec1,VECOPS_GLOBAL snde_coord *vec2,snde_index n)
{
  snde_coord curval;
  snde_coord accum=0.0f;
  snde_index idx;
  
  for (idx=0;idx < n; idx++) {
    curval = vec1[idx] - vec2[idx];
    accum += curval*curval;
  }
  return accum; 
}


// not actually used yet:
static VECOPS_INLINE snde_coord distsqgloballocalvecn(VECOPS_GLOBAL snde_coord *vec1,VECOPS_LOCAL snde_coord *vec2,snde_index n)
{
  snde_coord curval;
  snde_coord accum=0.0f;
  snde_index idx;
  
  for (idx=0;idx < n; idx++) {
    curval = vec1[idx] - vec2[idx];
    accum += curval*curval;
  }
  return accum; 
}


static VECOPS_INLINE void subvecvec3(const snde_coord *vec1,const snde_coord *vec2,snde_coord *out)
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then vec2 must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void subglobalvecvec3(OCL_GLOBAL_ADDR const snde_coord *vec1,const snde_coord *vec2,snde_coord *out)
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then vec2 must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void addcoordcoord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *out)
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then at least one of them must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] + vec2.coord[outidx];
    
  }
}


static VECOPS_INLINE void accumcoordcoord3(snde_coord3 vec1,snde_coord3 *out)
// NOTE: if vec1 is 2D coordinates in a 3D projective space,
// then it must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out->coord[outidx] += vec1.coord[outidx];    
  }
}


static VECOPS_INLINE void accumvecvec3(const snde_coord* vec1,snde_coord *out)
// NOTE: if vec1 is 2D coordinates in a 3D projective space,
// then it must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] += vec1[outidx];    
  }
}


static VECOPS_INLINE void addcoordcoord4proj(snde_coord4 vec1,snde_coord4 vec2,snde_coord4 *out)
// 3d coords in 4d projective space only
{
  int outidx;

  for (outidx=0;outidx < 4; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] + vec2.coord[outidx];
    
  }
}


static VECOPS_INLINE void subcoordcoord4(snde_coord4 vec1,snde_coord4 vec2,snde_coord4 *out)
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then at least one of them must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 4; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] - vec2.coord[outidx];
    
  }
}


static VECOPS_INLINE void subcoordcoord3(snde_coord3 vec1,snde_coord3 vec2,snde_coord3 *out)
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then at least one of them must be a vector, not a position
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

static VECOPS_INLINE void subvecvecn(snde_coord *vec1,snde_coord *vec2,snde_coord *out,snde_index n)
{
  snde_index outidx;

  for (outidx=0;outidx < n; outidx++) {
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
// NOTE: if vec1 and vec2 are 2D coordinates in a 3D projective space,
// then vec2 must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] + coeff* vec2.coord[outidx];
    
  }
}



static VECOPS_INLINE void addcoordscaledcoord4(snde_coord4 vec1,snde_coord coeff, snde_coord4 vec2,snde_coord4 *out)
// NOTE: if vec1 and vec2 are 3D coordinates in a 4D projective space,
// then vec2 must be a vector, not a position
{
  int outidx;

  for (outidx=0;outidx < 4; outidx++) {
    out->coord[outidx] = vec1.coord[outidx] + coeff* vec2.coord[outidx];
    
  }
}




static VECOPS_INLINE void normalize_wcoord4(snde_coord *vec)
/* operates in-place */
{
  vec[0] /= vec[3];
  vec[1] /= vec[3];
  vec[2] /= vec[3];
  vec[3] = 1.0f;
  
}



static VECOPS_INLINE snde_coord to_unit_vector4(snde_coord *vec)
/* operates in-place... returns scaling factor */
{
  snde_coord factor;

  factor=(snde_coord)(1.0f/sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]));
#ifdef __OPENCL_VERSION__
  /* if this is an opencl kernel, a W component makes the result invalid */
  if (vec[3] != 0.0f) {
    factor = snde_infnan(0); // NaN factor
  }
#else
  assert(vec[3]==0.0f); /* vectors should have no 'w' component */
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

  factor=(snde_coord)sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  return factor;
}

static VECOPS_INLINE snde_coord normsqvec3(const snde_coord *vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=(snde_coord)vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2];
  return factor;
}



static VECOPS_INLINE snde_coord normcoord3(snde_coord3 vec)
/* returns vector norm (3d non-projective coordinates) */
{
  snde_coord factor;

  factor=(snde_coord)sqrt(vec.coord[0]*vec.coord[0]+vec.coord[1]*vec.coord[1]+vec.coord[2]*vec.coord[2]);
  return factor;
}

static VECOPS_INLINE snde_coord normsqcoord3(snde_coord3 vec)
/* returns square of vector norm (3d non-projective coordinates) */
{
  snde_coord factor;

  factor=(vec.coord[0]*vec.coord[0]+vec.coord[1]*vec.coord[1]+vec.coord[2]*vec.coord[2]);
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
/* in-place vector normalization (3d non-projective coordinates) */
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

  factor=(snde_coord)sqrt(vec[0]*vec[0]+vec[1]*vec[1]);
  return factor;
}

static VECOPS_INLINE snde_coord normcoord2(snde_coord2 vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=(snde_coord)sqrt(vec.coord[0]*vec.coord[0]+vec.coord[1]*vec.coord[1]);
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

  factor=(snde_coord)(1.0f/sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]));
  vec[0] *= factor;
  vec[1] *= factor;
  vec[2] *= factor;

  return factor;
  
}

static VECOPS_INLINE snde_coord to_unit_coord3(snde_coord3 *vec)
/* operates in-place */
{
  snde_coord factor;

  factor=(snde_coord)(1.0f/sqrt(vec->coord[0]*vec->coord[0]+vec->coord[1]*vec->coord[1]+vec->coord[2]*vec->coord[2]));
  vec->coord[0] *= factor;
  vec->coord[1] *= factor;
  vec->coord[2] *= factor;

  return factor;
  
}


static VECOPS_INLINE void sign_nonzero3(snde_coord *input,snde_coord *output)
{
  int cnt;
  for (cnt=0;cnt < 3;cnt++) {
    if (input[cnt] < 0.0f) output[cnt]=-1.0f;
    else output[cnt]=1.0f;
  }
}

static VECOPS_INLINE void sign_nonzerocoord3(snde_coord3 input,snde_coord3 *output)
{
  int cnt;
  for (cnt=0;cnt < 3;cnt++) {
    if (input.coord[cnt] < 0.0f) output->coord[cnt]=-1.0f;
    else output->coord[cnt]=1.0f;
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
    out[cnt]=(snde_coord)((vec1[cnt]+vec2[cnt])/2.0f);
  }
}

static VECOPS_INLINE void mean2coord3(const snde_coord3 vec1,const snde_coord3 vec2,snde_coord3 *out)
{
  int cnt;
  
  for (cnt=0;cnt < 3; cnt++) {
    out->coord[cnt]=(vec1.coord[cnt]+vec2.coord[cnt])/2.0f;
  }
}

static VECOPS_INLINE snde_bool equalcoord3(const snde_coord3 vec1,const snde_coord3 vec2)
{
  return vec1.coord[0]==vec2.coord[0] && vec1.coord[1]==vec2.coord[1] && vec1.coord[2]==vec2.coord[2];
}

static VECOPS_INLINE snde_bool equalcoord4(const snde_coord4 vec1,const snde_coord4 vec2)
{
  return vec1.coord[0]==vec2.coord[0] && vec1.coord[1]==vec2.coord[1] && vec1.coord[2]==vec2.coord[2] && vec1.coord[3]==vec2.coord[3];
}

static VECOPS_INLINE void coord4_posn_from_coord3(const snde_coord3 vec, snde_coord4 *out)
{
  out->coord[0] = vec.coord[0];
  out->coord[1] = vec.coord[1];
  out->coord[2] = vec.coord[2];
  out->coord[3] = 1.0;
}

static VECOPS_INLINE void coord4_vec_from_coord3(const snde_coord3 vec, snde_coord4 *out)
{
  out->coord[0] = vec.coord[0];
  out->coord[1] = vec.coord[1];
  out->coord[2] = vec.coord[2];
  out->coord[3] = 0.0;
}

static VECOPS_INLINE void fmatrixsolve_print(snde_coord *A, snde_coord *b, size_t n, size_t nsolve,size_t *pivots)
{
  // print A and b
  int printrow,printcol;
  printf("A:\n");
  for (printrow=0; printrow < n; printrow++) {
    for (printcol=0;printcol < n; printcol++) {
      printf("%10f  ",(A[pivots[printrow] + printcol*n]));
    }
    printf("\n");
  }
  printf("b:\n");
  for (printrow=0; printrow < n; printrow++) {
    for (printcol=0;printcol < nsolve; printcol++) {
      printf("%10f  ",(b[pivots[printrow] + printcol*n]));
    }
    printf("\n");
  }
  
  printf("\n\n");
}


static VECOPS_INLINE void fmatrixmul(snde_coord *A1,snde_index A1_dimlen[2],snde_index A1_strides[2],
				     snde_coord *A2,snde_index A2_dimlen[2],snde_index A2_strides[2],
				     snde_coord *Aout,snde_index Aout_dimlen[2],snde_index Aout_strides[2])
{
  snde_index Aout_i1,Aout_i2,isum;
  if (A1_dimlen[1] != A2_dimlen[0] || A1_dimlen[0] != Aout_dimlen[0] || A2_dimlen[1] != Aout_dimlen[1]) {
    // mismatched dimensions
#ifdef __OPENCL_VERSION__
    // Can't do assert() in OpenCL, so we just set the output to all NaN's.
    for (Aout_i1=0;Aout_i1 < Aout_dimlen[0];Aout_i1++) {
      for (Aout_i2=0;Aout_i2 < Aout_dimlen[1];Aout_i2++) {
	Aout[Aout_i1 * Aout_strides[0] + Aout_i2 * Aout_strides[1]] = snde_infnan(0); // NaN
      }
    }
    return;
    
#else // __OPENCL_VERSION__
    assert(0); // matrix dimensions must agree
#endif
  }

  
  for (Aout_i1=0;Aout_i1 < Aout_dimlen[0];Aout_i1++) {
    for (Aout_i2=0;Aout_i2 < Aout_dimlen[1];Aout_i2++) {

      snde_coord accum=0.0f;
      
      for (isum=0; isum < A1_dimlen[1]; isum++) {
	accum += A1[ Aout_i1*A1_strides[0] + isum*A1_strides[1] ]*A2[ isum*A2_strides[0] + Aout_i2*A2_strides[1] ];
      }
      
      Aout[Aout_i1 * Aout_strides[0] + Aout_i2 * Aout_strides[1]] = accum;
      
    }
  }
}


static VECOPS_INLINE void fmatrixsolve(snde_coord *A,snde_coord *b,size_t n,size_t nsolve,size_t *pivots,int printflag)
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
    pivots[row]=row; // pivots[row] is the index of which physical row we should go to for conceptual row #row
                     // in the L, U triangular decomposition.
  }

  if (printflag) {
    //fmatrixsolve_print(A, b, n, nsolve,pivots);
  }
  
  for (row=0; row < n; row++) {
    // find largest magnitude row
    old_pivots_row=pivots[row];
    bestpivot=(snde_coord)fabs(A[pivots[row] + row*n]);  // pull out this diagonal etnry to start
    swapped=FALSE;
    for (rsrch=row+1; rsrch < n; rsrch++) {
      if (fabs(A[pivots[rsrch] + row*n]) > bestpivot) {
	bestpivot=(snde_coord)fabs(A[pivots[rsrch] + row*n]);
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
    
    if (printflag) {
      //fmatrixsolve_print(A, b, n, nsolve,pivots);
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

    if (printflag) {
      //fmatrixsolve_print(A, b, n, nsolve,pivots);
      //printf("Not printing the matrix!\n");
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


#ifdef VECOPS_DOUBLEPREC

static VECOPS_INLINE void fmatrixsolve_dp(double *A,double *b,size_t n,size_t nsolve,size_t *pivots,int printflag)
// solves A*x=b, where A is n*n, b is n*nsolve, and x is n*1
// must provide a n-length vector of size_t "pivots" that this routine uses for intermediate storage.
// *** NOTE: *** This routine will overwrite the contents of A and b... stores the
// result in b. 
// NOTE: A and b should be stored column major (Fortran style)
{
  size_t row,rsrch,col,succ_row,pred_row,rowcnt;
  double bestpivot,leading_val;
  size_t old_pivots_row;
  size_t solvecnt;
  double first_el,pred_val;
  int swapped;
  size_t swappedentry;
  
  // initialize blank pivots
  for (row=0; row < n; row++) {
    pivots[row]=row; // pivots[row] is the index of which physical row we should go to for conceptual row #row
                     // in the L, U triangular decomposition.
  }

  if (printflag) {
    //fmatrixsolve_print(A, b, n, nsolve,pivots);
  }
  
  for (row=0; row < n; row++) {
    // find largest magnitude row
    old_pivots_row=pivots[row];
    bestpivot=(double)fabs(A[pivots[row] + row*n]);  // pull out this diagonal etnry to start
    swapped=FALSE;
    for (rsrch=row+1; rsrch < n; rsrch++) {
      if (fabs(A[pivots[rsrch] + row*n]) > bestpivot) {
	bestpivot=(double)fabs(A[pivots[rsrch] + row*n]);
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
    
    if (printflag) {
      //fmatrixsolve_print(A, b, n, nsolve,pivots);
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

    if (printflag) {
      //fmatrixsolve_print(A, b, n, nsolve,pivots);
      //printf("Not printing the matrix!\n");
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

#endif // VECOPS_DOUBLEPREC

#endif // SNDE_VECOPS_H
