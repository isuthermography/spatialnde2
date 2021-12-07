#include "snde/rec_display_colormap.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/colormap.h"

namespace snde {
  template <typename T>
  class colormap_recording: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>
  {
  public:
    colormap_recording(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>(rss,inst)
    {
      
    }
    
    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::exec_function_override_type exec_function_override_type;
    
    // just using the default for decide_new_revision and compute_options
 
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> recording, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity,std::vector<snde_index> base_position,unsigned u_dim,unsigned v_dim) 
    {
      // define_recs code
      snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<ndtyped_recording_ref<snde_rgba>> result_rec = create_typed_recording_ref_math<snde_rgba>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=recording->rec->metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(recording->layout.dimlen,true); // Note fortran order flag -- required by renderer
	   
	  
	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    });
	  
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}
	    std::vector<snde_index> pos(base_position);
	    
	    // !!!*** should implement OpenCL version
	    // !!!*** OpenCL version must generate fortran-ordered
	    // output
	    for (snde_index vpos=0;vpos < recording->layout.dimlen.at(v_dim);vpos++){
	      for (snde_index upos=0;upos < recording->layout.dimlen.at(u_dim);upos++){
		pos.at(u_dim)=upos;
		pos.at(v_dim)=vpos;
		//result_rec->element(pos) = do_colormap(colormap_type,recording->element(pos)-offset)/unitsperintensity;
		result_rec->element(pos) = snde_colormap(colormap_type,(recording->element(pos)-offset)/unitsperintensity,255);
	      }
	    }
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready() 
	    result_rec->rec->mark_as_ready();
	  }); 
	});
      });
    }
    
    
  };


  std::shared_ptr<math_function> define_colormap_function()
  {
    return std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      // ***!!!! Should extend beyond just floating point types ***!!!
      // ***!!!! Should implement OpenCL acceleration
      return make_cppfuncexec_floatingtypes<colormap_recording>(rss,inst);
      
    }); 
  }


  static int registered_colormap_function = register_math_function(std::make_shared<registered_math_function>("spatialnde2.colormap",define_colormap_function));
  
  
  
};


