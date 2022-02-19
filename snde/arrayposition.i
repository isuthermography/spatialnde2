%{
  #include "arrayposition.hpp"
%}

namespace snde {
  class arraylayout {
  public:
    std::vector<snde_index> dimlen; // multidimensional shape...
    std::vector<snde_index> strides; // stride for each dimension... see numpy manual for detailed discussion. strides[0] being smallest is fortran layout; strides[last] being smallest is C layout
    
    arraylayout(std::vector<snde_index> dimlen,bool fortran_layout=false);
    arraylayout(std::vector<snde_index> dimlen,std::vector<snde_index> strides);
    bool is_c_contiguous();
    
    bool is_f_contiguous();
    
    bool is_contiguous();
    
    snde_index flattened_length();
    bool cachefriendly_indexing();

    /* nested classes not supported by swig...
    class arrayposition {
    public:
      // WARNING: Mutable
      std::shared_ptr<arraylayout> layout; // number of dimensions ndim is layout.dimlen.size()
      std::vector<snde_index> pos; // length ndim, unless ndim==0 in which case length 1
      snde_bool fortran_indexing;  // Not related to strides, which are physical layout. Instead indicates when we increment the index to increment the first element of pos, not the last element of pos. For efficient indexing, generally want fortran_indexing when you have fortran_layout
      
      arrayposition(const arraylayout &layout,bool fortran_indexing=false);
      arrayposition(const arraylayout &layout,snde_index intpos,bool fortran_indexing=false);
      arrayposition(const arraylayout &layout,std::vector<snde_index> pos,bool fortran_indexing=false);
      
      //arrayposition &operator++();
      //arrayposition &operator--();
    
      //arrayposition operator++(int dummy);
      //arrayposition operator--(int dummy);
      
      bool operator==(const arrayposition &other);
      bool operator!=(const arrayposition &other);
    };
    */
    
    //typedef arrayposition iterator;
    
    
    //iterator begin();
    
    //iterator end();    
    
    snde_index begin_flattened();
    snde_index end_flattened();
    
  };
  %extend arraylayout {
    std::string __str__()
    {
      std::string ret;
      size_t dimnum;
      
      ret+=snde::ssprintf("%d dimensional array: (",self->dimlen.size());
      
      for (dimnum=0;dimnum < self->dimlen.size(); dimnum++) {
	ret += snde::ssprintf("%llu",(unsigned long long)self->dimlen.at(dimnum));
	if (self->dimlen.size() >= 1 && dimnum < self->dimlen.size()-1) {
	  ret += " x ";
	}
      }
      ret += "); strides = (";
      for (dimnum=0;dimnum < self->dimlen.size(); dimnum++) {
	ret += snde::ssprintf("%llu",(unsigned long long)self->strides.at(dimnum));
	if (self->dimlen.size() >= 1 && dimnum < self->dimlen.size()-1) {
	  ret += ", ";
	}
      }
      ret += ")";
 
      if (self->is_c_contiguous()) {
	ret += "; c_contiguous";
      } else if (self->is_f_contiguous()) {
	ret += "; f_contiguous";
      } else if (self->is_contiguous()) {
	ret += "; contiguous";
      }
      ret += ".";
      return ret;
    }
  };

  %extend arraylayout {
    std::string __repr__()
    {
      std::string ret;
      size_t dimnum;
      
      ret+=snde::ssprintf("%d dimensional array: (",self->dimlen.size());
      
      for (dimnum=0;dimnum < self->dimlen.size(); dimnum++) {
	ret += snde::ssprintf("%llu",(unsigned long long)self->dimlen.at(dimnum));
	if (self->dimlen.size() >= 1 && dimnum < self->dimlen.size()-1) {
	  ret += " x ";
	}
      }
      ret += "); strides = (";
      for (dimnum=0;dimnum < self->dimlen.size(); dimnum++) {
	ret += snde::ssprintf("%llu",(unsigned long long)self->strides.at(dimnum));
	if (self->dimlen.size() >= 1 && dimnum < self->dimlen.size()-1) {
	  ret += ", ";
	}
      }
      ret += ")";
 
      if (self->is_c_contiguous()) {
	ret += "; c_contiguous";
      } else if (self->is_f_contiguous()) {
	ret += "; f_contiguous";
      } else if (self->is_contiguous()) {
	ret += "; contiguous";
      }
      ret += ".";
      return ret;
    }
  };
};

