#ifndef SNDE_COLORMAP_H
#define SNDE_COLORMAP_H

#ifndef __OPENCL_VERSION__
#include <errno.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

#include "snde/snde_types.h"
#endif

#ifdef _MSC_VER
#define COLORMAP_INLINE  __inline
#else
#define COLORMAP_INLINE  inline
#endif

#define SNDE_COLORMAP_GRAY 0
#define SNDE_COLORMAP_HOT 1
#define SNDE_COLORMAP_COLORFUL 2

static inline float snde_round_to_uchar(float inp)
{
  inp=round(inp);
  if (inp > 255.0f) {
    inp=255.0f;
  } else if (inp < 0.0f) {
    inp=0.0f;
  }
  return inp;
}


static inline snde_rgba snde_colormap(snde_index ColorMap,float val,uint8_t alpha)
{
  snde_rgba out;
  
  if (ColorMap==SNDE_COLORMAP_HOT) {
    float scaledval=val*3.0f;
    if (scaledval < 0.0f) scaledval=0.0f;
    if (scaledval > 3.0f) scaledval=3.0f;

    if (scaledval <= 1.0f) {
      out.r=(unsigned char)round(scaledval*255.0f);
      out.g=0;
      out.b=0;
      out.a=alpha;
    } else if (scaledval <= 2.0f) {
      out.r=255;
      out.g=(unsigned char)round((scaledval-1.0f)*255.0f);
      out.b=0;
      out.a=alpha;
    } else {
      out.r=255;
      out.g=255;
      out.b=(unsigned char)round((scaledval-2.0f)*255.0f);
      out.a=alpha;
    }
    
  } else if (ColorMap==SNDE_COLORMAP_COLORFUL) {
    float scaledval=val*4.0f;
    if (scaledval < 0.0f) scaledval=0.0f;
    if (scaledval > 4.0f) scaledval=4.0f;

    if (scaledval <= 0.5f) {
      out.r=0;
      out.g=0;
      out.b=(unsigned char)round((scaledval+0.5f)*255.0f);
      out.a=alpha;
    } else if (scaledval <= 1.5f) {
      out.r=0;
      out.g=(unsigned char)round((scaledval-0.5f)*255.0f);
      out.b=255;
      out.a=alpha;
    } else if (scaledval <= 2.5f) {
      out.r=(unsigned char)round((scaledval-1.5f)*255.0f);
      out.g=255;
      out.b=(unsigned char)round((2.5f-scaledval)*255.0f);
      out.a=alpha;
    } else if (scaledval <= 3.5f) {
      out.r=255;
      out.g=(unsigned char)round((3.5f-scaledval)*255.0f);
      out.b=0;
      out.a=alpha;
    } else {
      out.r=255;
      out.g=(unsigned char)round((scaledval-3.5f)*2.0f*255.0f);
      out.b=(unsigned char)round((scaledval-3.5f)*2.0f*255.0f);
      out.a=alpha;
      
    }
  } else {
    // SNDE_COLORMAP_GRAY
    out.r = (unsigned char)snde_round_to_uchar(val*255.0f);
    out.g = out.r;
    out.b = out.g;
    out.a = alpha;
  }
  return out;
}

#endif // SNDE_COLORMAP_H
