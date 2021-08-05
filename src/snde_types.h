#ifndef SNDE_TYPES_H
#define SNDE_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif


#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif


  typedef float snde_float32;
  typedef double snde_float64;
  
#ifdef SNDE_HAVE_FLOAT16
  typedef __fp16 snde_float16
#endif
  
#ifdef __cplusplus
}
#endif


#endif // SNDE_TYPES_H
