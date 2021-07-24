namespace snde {
  
  class math_parameter {
  public:
    unsigned paramtype; // SNDE_MFPT_XXX from wfmmath.hpp


    math_parameter(paramtype);
    // Rule of 3
    math_parameter(const math_parameter &) = delete;
    math_parameter& operator=(const math_parameter &) = delete; 
    virtual ~math_parameter()=default;  // virtual destructor required so we can be subclassed

    // default implementations that just raise runtime_error
    virtual std::string get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    virtual int64_t get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    virtual double get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    virtual std::shared_ptr<waveform> get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context); // should only return ready waveforms because we shouldn't be called until dependencies are ready

    // default implementations that returns an empty set
    virtual std::set<std::string> get_prerequisites(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the waveform
  };


  class math_parameter_string_const: public math_parameter {
  public:
    std::string string_constant;

    math_parameter_string_const(std::string string_constant);
    virtual std::string get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    
  };


  class math_parameter_int_const: public math_parameter {
  public:
    int64_t int_constant;

    math_parameter_int_const(int64_t int_constant);
    virtual int64_t get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    
  };

  class math_parameter_double_const: public math_parameter {
  public:
    double double_constant;

    math_parameter_double_const(double double_constant);
    virtual double get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context);
    
  };


  class math_parameter_waveform: public math_parameter {
  public:
    std::string channel_name;

    math_parameter_waveform(std::string channel_name);
    virtual std::shared_ptr<waveform> get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context); // should only return ready waveforms
    virtual std::set<std::string> get_prerequisites(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the waveform
    
  };

  // ***!!! Could have more classes here to implement e.g. parameters derived from metadata, expressions involving metadata, etc. 

};
