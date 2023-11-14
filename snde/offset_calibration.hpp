#ifndef SNDE_OFFSET_CALIBRATION_HPP
#define SNDE_OFFSET_CALIBRATION_HPP

namespace snde {
  class offset_calibration_single_point : public math_instance_parameter {
  public:
    std::string component_path[2]; // path to identify the relevant component within our scene graph.
    snde_coord3 local_coord_posn[2];

    offset_calibration_single_point(std::string component_path0,
				    snde_coord3 local_coord_posn0,
				    std::string component_path1,
				    snde_coord3 local_coord_posn1);
    
    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions

    virtual bool operator!=(const math_instance_parameter &ref);
    
  };
  
  std::shared_ptr<math_function> define_spatialnde2_offset_calibration_function();

  extern SNDE_API std::shared_ptr<math_function> offset_calibration_function;

};


#endif // SNDE_OFFSET_CALIBRATION_HPP
