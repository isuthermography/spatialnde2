#include "wfmmath.hpp"


namespace snde {
  math_function::math_function(size_t num_results,const std::list<std::tuple<std::string,unsigned>> &param_names_types) :
    num_results(num_results),
    param_names_types(param_names_types)
  {
    new_revision_optional=false;
    pure_optionally_mutable=false;
    mandatory_mutable=false;
    self_dependent=false; 
  }

  virtual std::shared_ptr<instantiated_math_function> instantiated_math_function::clone()
  {
    // This code needs to be repeated in every derived class with the make_shared<> referencing the derived class
    std::shared_ptr<instantiated_math_function> copy = std::make_shared<instaniated_math_function>(*this);

    // see comment at start of definition of class instantiated_math_function
    if (definition) {
      assert(!copy->original_function);
      copy->original_function = definition;
      copy->definition = nullptr; 
    }
    return copy;
  }

  

};
