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

snde_coord my_infnan(int error) /* be sure to disable SIGFPE */
{
  
  if (error==ERANGE) return *((my_infnan_float32_ptr_t)&Infconst);
  else if (error==-ERANGE) return *((my_infnan_float32_ptr_t)&NegInfconst);
  else return *((my_infnan_float32_ptr_t)&NaNconst);
}



static VECOPS_INLINE void multmat23vec(snde_coord *mat,snde_coord *vec,snde_coord *out)
/* Multiply 2x3 matrix by 3-vector, giving 2-vector */
{
  int outel,sumidx;

  for (outel=0;outel < 2; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 3; sumidx++) {
      out[outel] = out[outel] + mat[ outel*3 + sumidx]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multvecmat23(snde_coord *vec,snde_coord *mat,snde_coord *out)
/* Multiply 2-vector by 2x3 matrix giving 3-vector  */
{
  int outel,sumidx;

  for (outel=0;outel < 3; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 2; sumidx++) {
      out[outel] = out[outel] + mat[ sumidx*3 + outel]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multmatvec4(snde_coord *mat,snde_coord *vec,snde_coord *out)
{
  int outel,sumidx;

  for (outel=0;outel < 4; outel++) {
    out[outel]=0.0;
    for (sumidx=0;sumidx < 4; sumidx++) {
      out[outel] = out[outel] + mat[ outel*4 + sumidx]*vec[sumidx];
    }
  }
}

static VECOPS_INLINE void multmatvec3(snde_coord *mat,snde_coord *vec,snde_coord *out)
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

static VECOPS_INLINE snde_coord dotvecvec2(snde_coord *vec1,snde_coord *vec2)
{
  int sumidx;
  snde_coord val=0.0;
  for (sumidx=0;sumidx < 2; sumidx++) {
    val = val + vec1[sumidx]*vec2[sumidx];
    
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


static VECOPS_INLINE void scalevec2(snde_coord coeff,snde_coord *vec1,snde_coord *out)
{
  size_t cnt;
  for (cnt=0;cnt < 2; cnt++) {
    out[cnt]=coeff*vec1[cnt];
  }
}



static VECOPS_INLINE void subvecvec3(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void subvecvec2(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 2; outidx++) {
    out[outidx] = vec1[outidx] - vec2[outidx];
    
  }
}

static VECOPS_INLINE void addvecscaledvec3(snde_coord *vec1,snde_coord coeff, snde_coord *vec2,snde_coord *out)
{
  int outidx;

  for (outidx=0;outidx < 3; outidx++) {
    out[outidx] = vec1[outidx] + coeff* vec2[outidx];
    
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

static VECOPS_INLINE void normalizevec3(snde_coord *vec)
/* in-place vector normalization */
{
  snde_coord factor;

  factor=normvec3(vec);
  vec[0] /= factor;
  vec[1] /= factor;
  vec[2] /= factor;
}


static VECOPS_INLINE snde_coord normvec2(snde_coord *vec)
/* returns vector norm */
{
  snde_coord factor;

  factor=sqrt(vec[0]*vec[0]+vec[1]*vec[1]);
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


static VECOPS_INLINE void sign_nonzero3(snde_coord *input,snde_coord *output)
{
  int cnt;
  for (cnt=0;cnt < 3;cnt++) {
    if (input[cnt] < 0.0) output[cnt]=-1.0;
    else output[cnt]=1.0;
  }
}


static VECOPS_INLINE void multvecvec3(snde_coord *vec1,snde_coord *vec2,snde_coord *output)
{
  int cnt;
  for (cnt=0;cnt < 3; cnt++) {
    output[cnt]=vec1[cnt]*vec2[cnt];
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


static VECOPS_INLINE void mean2vec3(snde_coord *vec1,snde_coord *vec2,snde_coord *out)
{
  int cnt;
  
  for (cnt=0;cnt < 3; cnt++) {
    out[cnt]=(vec1[cnt]+vec2[cnt])/2.0;
  }
}
