#ifndef SNDE_ARRAYPOSITION_HPP
#define SNDE_ARRAYPOSITION_HPP

#include "snde/array_index_ops.h"

#include <vector>

namespace snde {
  class arraylayout {
  public:
    std::vector<snde_index> dimlen; // multidimensional shape...
    std::vector<snde_index> strides; // stride for each dimension... see numpy manual for detailed discussion. strides[0] being smallest is fortran layout; strides[last] being smallest is C layout

    arraylayout() 
    {

    }
    
    arraylayout(std::vector<snde_index> dimlen,bool fortran_layout=false) :
      dimlen(dimlen),
      strides(dimlen.size(),0)
    {
      snde_index dimnum;
      snde_index ndim=this->dimlen.size();
      snde_index prodsz=1;
      
      if (fortran_layout) {
	for (dimnum=0; dimnum < this->dimlen.size();dimnum++) {
	  this->strides[dimnum]=prodsz;
	  prodsz *= this->dimlen[dimnum];
	}
	
      } else {
	for (dimnum=0; dimnum < this->dimlen.size();dimnum++) {
	  this->strides[ndim-dimnum-1]=prodsz;
	  prodsz *= this->dimlen[ndim-dimnum-1];
	}
      }
    }
    
    arraylayout(std::vector<snde_index> dimlen,std::vector<snde_index> strides) :
      dimlen(dimlen),
      strides(strides)
    {
      
    }

    arraylayout(const arraylayout &) = default; // default copy constructor
    arraylayout &operator=(const arraylayout &) = default; // default copy assignment
    ~arraylayout() = default; // default destructor

    bool operator==(const arraylayout &other)
    {
      return dimlen==other.dimlen && strides==other.strides;
    }

    bool operator!=(const arraylayout &other)
    {
      return !(*this == other);
    }

    bool is_c_contiguous()
    {
      return (bool)snde_array_is_c_contiguous(dimlen.data(),strides.data(),dimlen.size());
    }
    
    bool is_f_contiguous()
    {
      return (bool)snde_array_is_f_contiguous(dimlen.data(),strides.data(),dimlen.size());
    }
    
    bool is_contiguous()
    {
      std::vector<snde_index> workbuf(dimlen.size());
      return (bool)snde_array_is_contiguous(dimlen.data(),strides.data(),workbuf.data(),dimlen.size());
    }
    
    snde_index flattened_length() // total number of elements
    {
      return snde_array_flattened_length(dimlen.data(),dimlen.size());
    }

    snde_index flattened_size() // number of contiguous elements that need to be stored/transfered (different from flattened length if the data has holes due to strides, etc.
    {
      assert(dimlen.size()==strides.size());
      return snde_array_flattened_size(dimlen.data(),strides.data(),strides.size());      
    }
      
    bool cachefriendly_indexing() // returns true for Fortran mode
    {
      return (bool)snde_array_cachefriendly_indexing(strides.data(),dimlen.size()); 
    }
    
    class arrayposition {
    public:
      // WARNING: Mutable
      std::shared_ptr<arraylayout> layout; // number of dimensions ndim is layout.dimlen.size()
      std::vector<snde_index> pos; // length ndim, unless ndim==0 in which case length 1
      snde_bool fortran_indexing;  // Not related to strides, which are physical layout. Instead indicates when we increment the index to increment the first element of pos, not the last element of pos. For efficient indexing, generally want fortran_indexing when you have fortran_layout
      
      arrayposition(const arraylayout &layout,bool fortran_indexing=false) :
	layout(std::make_shared<arraylayout>(layout)),
	pos(std::max(layout.dimlen.size(),(size_t)1),0),
	fortran_indexing(fortran_indexing)
      {
	if (this->layout->dimlen.size()==0) {
	  pos.push_back(0); // 0-dim array always has 1 element in pos
	}
      }
      
      arrayposition(const arraylayout &layout,snde_index intpos,bool fortran_indexing=false) :
	layout(std::make_shared<arraylayout>(layout)),
	pos(std::max(layout.dimlen.size(),(size_t)1),0),
	fortran_indexing(fortran_indexing)
      {
	snde_array_index_from_integer(this->layout->dimlen.data(),
				      this->layout->dimlen.size(),
				      (snde_bool)fortran_indexing,
				      intpos,
				      pos.data());
      }
      
      arrayposition(const arraylayout &layout,std::vector<snde_index> pos,bool fortran_indexing=false) :
	layout(std::make_shared<arraylayout>(layout)),
	pos(pos),
	fortran_indexing(fortran_indexing)
      {
	assert((layout.dimlen.size() == 0  &&  pos.size() == 1) || (layout.dimlen.size() == pos.size()));
	assert(pos.size() >= 1);
      }
      
      arrayposition &operator++()
      // pre-increment
      {
	snde_array_index_increment(layout->dimlen.data(),
				   layout->dimlen.size(),
				   (snde_bool)fortran_indexing,
				   pos.data());
	return *this;
      }
      
      arrayposition &operator--()
      // pre-decrement
      {
	snde_array_index_decrement(layout->dimlen.data(),
				   layout->dimlen.size(),
				   (snde_bool)fortran_indexing,
				   pos.data());
	return *this;
      }
      
    
      arrayposition operator++(int dummy)
      // post-increment
      {
	arrayposition copy=*this; 
	snde_array_index_increment(layout->dimlen.data(),
				   layout->dimlen.size(),
				   (snde_bool)fortran_indexing,
				   pos.data());
	return copy;
      }
      
      arrayposition operator--(int dummy)
      // post-decrement
      {
	arrayposition copy=*this; 
	snde_array_index_decrement(layout->dimlen.data(),
				   layout->dimlen.size(),
				   (snde_bool)fortran_indexing,
				   pos.data());
	return copy;
      }
      
      bool operator==(const arrayposition &other)
      {
	//snde_index dimnum;
	assert(layout->dimlen.size()==other.layout->dimlen.size());
	
	return (bool)snde_array_index_equal(layout->dimlen.data(),
					    pos.data(),
					    other.pos.data(),
					    layout->dimlen.size());
      }
      bool operator!=(const arrayposition &other)
      {
      return !(*this==other);
      }
      
    };
    
    typedef arrayposition iterator;
    
    
    iterator begin()
    {
      return arrayposition(*this,cachefriendly_indexing());
    }
    
    iterator end()
    {
      return arrayposition(*this,flattened_length(),cachefriendly_indexing());
    }
    
    
    snde_index begin_flattened()
    {
      return 0;
    }
    
    snde_index end_flattened()
    {
      return flattened_length();
    }
    
  };
  
};

#endif // SNDE_ARRAYPOSITION_HPP
