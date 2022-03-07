#ifndef SNDE_TYPES_H
#define SNDE_TYPES_H

// CMake's CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS is set, but some global variables still require 
// explicit import and export flags to compile properly with MSVC and other compilers that
// behave similarly.  Newer versions of GCC shouldn't care about the presence of dllimport
// or dllexport, but it doesn't need it.
#ifdef _WIN32
#ifdef SPATIALNDE2_SHAREDLIB_EXPORT
#define SNDE_API __declspec(dllexport)
#else
#define SNDE_API __declspec(dllimport)
#endif
#else
#define SNDE_API
#endif


#ifdef _WIN32
#ifdef SPATIALNDE2_OCL_SHAREDLIB_EXPORT
#define SNDE_OCL_API __declspec(dllexport)
#else
#define SNDE_OCL_API __declspec(dllimport)
#endif
#else
#define SNDE_OCL_API
#endif



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
#ifdef SNDE_OCL_HAVE_DOUBLE
typedef double snde_float64;
#endif // SNDE_OCL_HAVE_DOUBLE
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

#define SIZEOF_SNDE_INDEX 8 // must be kept consistent with typedefs!
  
#define SNDE_INDEX_INVALID (~((snde_index)0))


  typedef struct _snde_complexfloat32 {
    snde_float32 real;
    snde_float32 imag;
  } snde_complexfloat32;

#if (!defined(__OPENCL_VERSION__) || defined(SNDE_OCL_HAVE_DOUBLE))
  typedef struct _snde_complexfloat64 {
    snde_float64 real;
    snde_float64 imag;
  } snde_complexfloat64;
#endif // (!defined(__OPENCL_VERSION__) || defined(SNDE_OCL_HAVE_DOUBLE))
  
#ifdef SNDE_HAVE_FLOAT16
  typedef struct _snde_complexfloat16 {
    snde_float16 real;
    snde_float16 imag;
  } snde_complexfloat16;

#endif // SNDE_HAVE_FLOAT16

  // interoperability between opencl and C/C++ code
#ifdef __OPENCL_VERSION__
#define OCL_GLOBAL_ADDR __global
#define OCL_LOCAL_ADDR __local
#define OCL_KERNEL __kernel
#else
#define OCL_GLOBAL_ADDR
#define OCL_LOCAL_ADDR
#define OCL_KERNEL
#endif
  
  
#ifdef __cplusplus
}
#endif


#endif // SNDE_TYPES_H
