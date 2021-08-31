%{
  #include "arrayposition.hpp"
%}

namespace snde {
  class arraylayout {
  public:
    snde_index base_index;
    std::vector<snde_index> dimlen; // multidimensional shape...
    std::vector<snde_index> strides; // stride for each dimension... see numpy manual for detailed discussion. strides[0] being smallest is fortran layout; strides[last] being smallest is C layout
    
    arraylayout(std::vector<snde_index> dimlen,bool fortran_layout=false,snde_index base_index=0);
    arraylayout(std::vector<snde_index> dimlen,std::vector<snde_index> strides,snde_index base_index=0);
    bool is_c_contiguous();
    
    bool is_f_contiguous();
    
    bool is_contiguous();
    
    snde_index flattened_length();
    bool cachefriendly_indexing();
    
    class arrayposition {
    public:
      // WARNING: Mutable
      std::shared_ptr<arraylayout> layout; // number of dimensions ndim is layout.dimlen.size()
      std::vector<snde_index> pos; // length ndim, unless ndim==0 in which case length 1
      snde_bool fortran_indexing;  // Not related to strides, which are physical layout. Instead indicates when we increment the index to increment the first element of pos, not the last element of pos. For efficient indexing, generally want fortran_indexing when you have fortran_layout
      
      arrayposition(const arraylayout &layout,bool fortran_indexing=false);
      arrayposition(const arraylayout &layout,snde_index intpos,bool fortran_indexing=false);
      arrayposition(const arraylayout &layout,std::vector<snde_index> pos,bool fortran_indexing=false);
      
      arrayposition &operator++();
      arrayposition &operator--();
    
      arrayposition operator++(int dummy);
      arrayposition operator--(int dummy);
      
      bool operator==(const arrayposition &other);
      bool operator!=(const arrayposition &other);
    };
    
    typedef arrayposition iterator;
    
    
    iterator begin();
    
    iterator end();    
    
    snde_index begin_flattened();
    snde_index end_flattened();
    
  };
  
};

#endif // SNDE_ARRAYPOSITION_HPP
