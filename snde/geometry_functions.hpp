#ifndef SNDE_GEOMETRY_FUNCTIONS_HPP
#define SNDE_GEOMETRY_FUNCTIONS_HPP

namespace snde {

  /* **** is the geometry_function class really needed? */
  class geometry_function {
  public:
    
    geometry_function(const geometry_function &)=delete; /* copy constructor disabled */
    geometry_function& operator=(const geometry_function &)=delete; /* copy assignment disabled */
    
    std::shared_ptr<trm_dependency> function;


    ~geometry_function()
    {
      
    }
  };

  std::shared_ptr<geometry_function> normals_function(std::shared_ptr<part>)
  {
    assert(!part->normals);
    
  }

  std::shared_ptr<geometry_function> inplanemat_function(std::shared_ptr<part>)
  {
    
  }

  std::shared_ptr<geometry_function> curvature_function(std::shared_ptr<part>)
  {
    
  }

  std::shared_ptr<geometry_function> boxes_function(std::shared_ptr<part>)
  {
    
  }

}

#endif // SNDE_GEOMETRY_FUNCTIONS_HPP
