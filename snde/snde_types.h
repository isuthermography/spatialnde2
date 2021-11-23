#ifndef SNDE_TYPES_H
#define SNDE_TYPES_H


#ifndef __OPENCL_VERSION__
// if this is not an OpenCL kernel
#include <assert.h>
#include <stdint.h>
#include <string.h>

#endif

#ifdef __cplusplus
extern "C" {
#endif


#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif


#ifdef __OPENCL_VERSION__
typedef double snde_float64;
typedef float snde_float32;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;

typedef char int8_t;
typedef int int32_t;
typedef short int16_t;


typedef ulong snde_index;
typedef uint snde_shortindex;
typedef long snde_ioffset;
typedef unsigned char snde_bool;

  
#else
  typedef float snde_float32;
  typedef double snde_float64;
  
#ifdef SNDE_HAVE_FLOAT16
  typedef __fp16 snde_float16
#endif

  typedef uint32_t snde_shortindex;
typedef unsigned char snde_bool;

  // Don't specify 64 bit integers in terms of int64_t/uint64_t to work around
  // https://github.com/swig/swig/issues/568
  //typedef uint64_t snde_index;
  //typedef int64_t snde_ioffset;
#ifdef SIZEOF_LONG_IS_8
  typedef unsigned long snde_index;
  typedef long snde_ioffset;
#else
  typedef unsigned long long snde_index;
  typedef long long snde_ioffset;
#endif


#endif // __OPENCL_VERSION__
  
#ifdef __cplusplus
}
#endif


#endif // SNDE_TYPES_H
