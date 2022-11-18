#ifndef SNDE_ARRAY_INDEX_OPS_H
#define SNDE_ARRAY_INDEX_OPS_H


#ifdef __OPENCL_VERSION__
#define SNDE_AIO_GLOBAL __global
#else
#define SNDE_AIO_GLOBAL

#include <stdlib.h> // for qsort

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#endif // __OPENCL_VERSION__

#define SNDE_AIO_STATIC_INLINE static inline

void SNDE_AIO_STATIC_INLINE
snde_array_index_from_integer(
			      SNDE_AIO_GLOBAL snde_index *dimlen,
			      snde_index ndim,
			      snde_bool fortran_indexing,
			      snde_index intpos,
			      snde_index *pos) //out 
// obtain index position from an integer offset
// pos always has at least one element:
// if ndim==0, pos[0]==0 references the one element vs
// pos[0]==SNDE_INDEX_INVALID references overflow/array end
{
  // overwrites pos
  snde_index dimnum; 

  pos[0]=0;
  
  if (fortran_indexing) {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      pos[dimnum] = intpos % dimlen[dimnum];
      intpos /= dimlen[dimnum];
    }

  } else {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      pos[ndim-dimnum-1] = intpos % dimlen[ndim-dimnum-1];
      intpos /= dimlen[ndim-dimnum-1];
    }
  }

  if (intpos > 0) {
    pos[0]=SNDE_INDEX_INVALID; // mark as overflow
  }

}





void SNDE_AIO_STATIC_INLINE
snde_array_index_increment(
			   SNDE_AIO_GLOBAL snde_index *dimlen,
			   snde_index ndim,
			   snde_bool fortran_indexing,
			   snde_index *pos) // inout
// increment index position
// pos always has at least one element:
{
  // overwrites pos
  snde_index dimnum; 
  
  if (fortran_indexing) {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      pos[dimnum]++;
      if (pos[dimnum] >= dimlen[dimnum]) {
	pos[dimnum] = 0;
      } else {
	break;
      }
    }
    
  } else {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      pos[ndim-dimnum-1]++;
      if (pos[ndim-dimnum-1] >= dimlen[ndim-dimnum-1]) {
	pos[ndim-dimnum-1] = 0;
      } else {
	break;
      }
    }
  }
  if (dimnum==ndim) {
    pos[0]=SNDE_INDEX_INVALID; // mark as overflow
  }
}


void SNDE_AIO_STATIC_INLINE
snde_array_index_decrement(
			   SNDE_AIO_GLOBAL snde_index *dimlen,
			   snde_index ndim,
			   snde_bool fortran_indexing,
			   snde_index *pos) //inout
// decrement index position
// pos always has at least one element:
{
  // overwrites pos
  snde_index dimnum;

  // treat invalid value as 1 past end, so decrement
  // to final element
  if (pos[0]==SNDE_INDEX_INVALID) {
    pos[0]=0;
    for (dimnum=0; dimnum < ndim;dimnum++) {
      pos[dimnum]=dimlen[dimnum]-1;
    }
    return;
  }
  
  if (fortran_indexing) {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      if (pos[dimnum] > 0) {
	pos[dimnum]--;
	break;
      } else {
	pos[dimnum] = dimlen[dimnum]-1;
      }
    }
    
  } else {
    for (dimnum=0; dimnum < ndim;dimnum++) {
      if (pos[ndim-dimnum-1] > 0) {
	pos[ndim-dimnum-1]--;
	break;
      } else {
	pos[ndim-dimnum-1] = dimlen[ndim-dimnum-1]-1;
      }
    }
  }
  if (dimnum==ndim) {
    pos[0]=SNDE_INDEX_INVALID; // mark as underflow
  }
}


snde_index SNDE_AIO_STATIC_INLINE
snde_array_index_rawval(
			SNDE_AIO_GLOBAL snde_index *strides,
			snde_index *pos, // in
			snde_index ndim,
			snde_index base_index)
// low-level index into memory from index position
{
  snde_index dimnum;
  snde_index accum;

  accum = base_index;
  for (dimnum=0;dimnum < ndim; dimnum++) {
    accum+=strides[dimnum]*pos[dimnum];
  }

  return accum;
}



snde_bool SNDE_AIO_STATIC_INLINE
snde_array_index_equal(
		       SNDE_AIO_GLOBAL const snde_index *dimlen,
		       const snde_index *pos1,
		       const snde_index *pos2,
		       snde_index ndim)
// specify if two indexes are equal (invalids match invalids)
// pos1, pos2 always at least length 1
{
  snde_index dimnum;
  snde_bool matching=TRUE;

  
  if (pos1[0]==SNDE_INDEX_INVALID && pos2[0]==SNDE_INDEX_INVALID) {
    return TRUE;
  }

  for (dimnum=0; dimnum < ndim; dimnum++) {
    if (pos1[dimnum] != pos2[dimnum]) {
      matching = FALSE;
      break;
    }
  }

  return matching;
}

snde_index SNDE_AIO_STATIC_INLINE
snde_array_flattened_length(
			    SNDE_AIO_GLOBAL snde_index *dimlen,
			    snde_index ndim)
{
  snde_index length=1;
  snde_index dimnum;

  for (dimnum=0;dimnum < ndim;dimnum++) {
    length *= dimlen[dimnum];
  }

  return length;
}


snde_index SNDE_AIO_STATIC_INLINE
snde_array_flattened_size(
			    SNDE_AIO_GLOBAL snde_index *dimlen,
			    SNDE_AIO_GLOBAL snde_index *strides,
			    snde_index ndim)
  // number of contiguous elements that need to be stored/transfered (different from flattened length if the data has holes due to strides, etc.
{
  snde_index last_element=0;
  snde_index dimnum;

  for (dimnum=0;dimnum < ndim;dimnum++) {
    //assert(dimlen[dimnum] > 0);
    if (dimlen[dimnum]==0) {
      return 0;
    }
    last_element += strides[dimnum]*(dimlen[dimnum]-1);
  }

  return last_element+1;
}


snde_bool SNDE_AIO_STATIC_INLINE
snde_array_cachefriendly_indexing( // returns non-zero for fortran mode
			    SNDE_AIO_GLOBAL snde_index *strides,
			    snde_index ndim)
{
  if (ndim==0) return FALSE;
  if (strides[0] < strides[ndim-1]) return TRUE;
  return FALSE;
}


snde_bool SNDE_AIO_STATIC_INLINE
snde_array_is_c_contiguous(SNDE_AIO_GLOBAL snde_index *dimlen,
			 SNDE_AIO_GLOBAL snde_index *strides,
			 snde_index ndim)
{
  snde_index dimnum;
  snde_index exp_stride=1; // expected stride
  for (dimnum=0; dimnum < ndim;dimnum ++) {
    if (strides[ndim-dimnum-1] != exp_stride) {
      return FALSE;
    }
    exp_stride *= dimlen[ndim-dimnum-1];
  }
  return TRUE;
}

snde_bool SNDE_AIO_STATIC_INLINE
snde_array_is_f_contiguous(SNDE_AIO_GLOBAL snde_index *dimlen,
			   SNDE_AIO_GLOBAL snde_index *strides,
			   snde_index ndim)
{
  snde_index dimnum;
  snde_index exp_stride=1; // expected stride
  for (dimnum=0; dimnum < ndim;dimnum ++) {
    if (strides[dimnum] != exp_stride) {
      return FALSE;
    }
    exp_stride *= dimlen[dimnum];
  }
  return TRUE;
}


#ifndef __OPENCL_VERSION__ // ignore under OpenCL because we don't have qsort there and probably don't need this

SNDE_AIO_STATIC_INLINE int snde_aic_stride_compare(const void *stride1_vp,const void *stride2_vp)
{
  snde_index *stride1;
  snde_index *stride2;

  stride1 = (snde_index *)stride1_vp;
  stride2 = (snde_index *)stride2_vp;
  if (*stride1==*stride2) return 0;
  if (*stride1 < *stride2) return -1;
  return 1;
}

SNDE_AIO_STATIC_INLINE int snde_aic_stride_compare_reverse(const void *stride1_vp,const void *stride2_vp)
{
  snde_index *stride1;
  snde_index *stride2;

  stride1 = (snde_index *)stride1_vp;
  stride2 = (snde_index *)stride2_vp;
  if (*stride1==*stride2) return 0;
  if (*stride1 > *stride2) return -1;
  return 1;
}



snde_bool SNDE_AIO_STATIC_INLINE
snde_array_is_contiguous(SNDE_AIO_GLOBAL snde_index *dimlen,
			 SNDE_AIO_GLOBAL snde_index *strides,
			 snde_index *workbuf, // length ndim
			 snde_index ndim)
{
  snde_index dimnum;
  snde_index exp_stride=1; // expected stride
  snde_bool f_contiguous,c_contiguous;
  
  // after strides sorted, array
  // should appear fortran-contiguous or c-contiguous (does this actually capture all possible contiguous cases???)
  for (dimnum=0;dimnum < ndim;dimnum++) {
    workbuf[dimnum]=strides[dimnum];
  }
  qsort(workbuf,ndim,sizeof(snde_index),&snde_aic_stride_compare);
  f_contiguous = snde_array_is_f_contiguous(dimlen,workbuf,ndim);

  // need to reverse-sort to test for c contiguity
  qsort(workbuf,ndim,sizeof(snde_index),&snde_aic_stride_compare_reverse);

  c_contiguous = snde_array_is_c_contiguous(dimlen,workbuf,ndim);

  return f_contiguous || c_contiguous;
  

}
#endif // !__OPENCL_VERSION__

#endif // ARRAYA_INDEX_OPS_H
