#include "snde/rec_display_colormap.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"




namespace snde {
  struct bpc_compare_pixel_coords{
    bool operator()(const std::pair<uint32_t,uint32_t>& A,const std::pair<uint32_t,uint32_t>& B) const {
      // return true if A is < B
      if (A.first < B.first) {
        return true;
      }
      if (A.first == B.first) {
        if (A.second < B.second) {
          return true;
        }
      }
      return false;
    }
  };
  
    
  template <typename T>
  class bad_pixel_correction: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>>
  {
  public:
    bad_pixel_correction(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>>(rss,inst)
    {
      
    }

        // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::shared_ptr<ndtyped_recording_ref<uint32_t>>>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::shared_ptr<ndtyped_recording_ref<uint32_t>>>::exec_function_override_type exec_function_override_type;
    
    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<T>> rawimage,std::shared_ptr<ndtyped_recording_ref<uint32_t>> bad_pixel_array) 
    {
      snde_index numdatapoints = rawimage->layout.dimlen.at(0)*rawimage->layout.dimlen.at(1);
      snde_index numbadpixels = bad_pixel_array->layout.dimlen.at(0);

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numdatapoints*2*sizeof(T), // data_bytes for transfer
							numbadpixels*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }


 
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> rawimage,std::shared_ptr<ndtyped_recording_ref<uint32_t>> bad_pixel_array) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      rawimage->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<ndtyped_recording_ref<T>> result_rec = create_typed_ndarray_ref_math<T>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,rawimage,bad_pixel_array ] () {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=rawimage->rec->metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,rawimage,bad_pixel_array ]() {
	  // lock_alloc code
	  result_rec->rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)
	  
	  
	  result_rec->allocate_storage({rawimage->layout.dimlen.at(0),rawimage->layout.dimlen.at(1)},rawimage->layout.is_f_contiguous()); // Note fortran order flag 
	 

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { rawimage, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
              { bad_pixel_array, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    },false);
	  
	  
	  return std::make_shared<exec_function_override_type>([ this, locktokens,result_rec,rawimage,bad_pixel_array ]() {
	    // exec code
	  
            
            // Proposed algorithm .Provide a list of pixel coordinates. For each bad pixel create a list of neighbors leaving out any that are off the image. Collect the list of bad pixels into a map. For each neighbor of  pixel check if the neighbor is a bad pixel and remove it from the list if it is. The result is a map from bad pixel coordinates to a list of valid neighbors. Then iterate through the list and correct the pixels. 

            // map indexed by bad pixel location that looks up a set of neighbors
            std::map<std::pair<uint32_t,uint32_t>,std::set<std::pair<uint32_t, uint32_t>,bpc_compare_pixel_coords>,bpc_compare_pixel_coords> neighbors_by_bad_pixel;
            
            for (snde_index bpnum=0; bpnum < bad_pixel_array->layout.dimlen.at(0);bpnum++) {
              uint32_t col,row;
              std::map<std::pair<uint32_t,uint32_t>,std::set<std::pair<uint32_t, uint32_t>,bpc_compare_pixel_coords>,bpc_compare_pixel_coords>::iterator badpixel_it;
              
              bool added;
              col = bad_pixel_array->element({bpnum,0});
              row = bad_pixel_array->element({bpnum,1});

              std::set<std::pair<uint32_t, uint32_t>,bpc_compare_pixel_coords> neighbors;
              if (col > 0) {
                neighbors.emplace(std::make_pair(col-1,row)); //pixel to left
              }
              if (col < rawimage->layout.dimlen.at(0)-1) {
                neighbors.emplace(std::make_pair(col+1,row)); //pixel to right
              }
              if (row > 0) {
                neighbors.emplace(std::make_pair(col,row-1)); //pixel to up
              }
              if (row < rawimage->layout.dimlen.at(1)-1) {
                neighbors.emplace(std::make_pair(col,row+1)); //pixel to down
              }
              std::tie(badpixel_it,added)=neighbors_by_bad_pixel.emplace(std::make_pair(col,row),neighbors);
            }

            // Now go through each bad pixel again and remove any neighbors that are themselves bad

            for (auto && badpixel_neighbors: neighbors_by_bad_pixel) {
              auto neighbor_next_it=badpixel_neighbors.second.end();
              for (auto neighbor_it=badpixel_neighbors.second.begin();neighbor_it != badpixel_neighbors.second.end(); neighbor_it=neighbor_next_it) {
                neighbor_next_it=neighbor_it;
                neighbor_next_it++;

                auto badpixel_it=neighbors_by_bad_pixel.find(*neighbor_it);
                if (badpixel_it != neighbors_by_bad_pixel.end()) {
                  // this pixel is one of our bad pixels
                  badpixel_neighbors.second.erase(neighbor_it);
                }
              }
            }

            if (result_rec->layout != rawimage->layout) {
              throw snde_error("bad pixel correction:source and results layouts must match");
              
            }

            // Copy the raw image into the result_recording
            memcpy(result_rec->shifted_arrayptr(),rawimage->shifted_arrayptr(),result_rec->layout.flattened_size()*sizeof(T));

            //then go through each pixel, average the neighbors and replace the badpixels in our output with the average.
            
            for (auto && badpixel_neighbors: neighbors_by_bad_pixel) {
              double sum = 0.0;
              size_t n=0;
              uint32_t badpixel_x, badpixel_y;
              std::tie(badpixel_x,badpixel_y)=badpixel_neighbors.first;
              
              for (auto neighbor_it=badpixel_neighbors.second.begin();neighbor_it != badpixel_neighbors.second.end(); neighbor_it++) {
                uint32_t neighbor_x;
                uint32_t neighbor_y;
                std::tie(neighbor_x,neighbor_y) = *neighbor_it;
                sum += (double) rawimage->element({neighbor_x,neighbor_y});
                n++;
              }
              result_rec->element({badpixel_x,badpixel_y}) = (T) (sum/n);
            }
            
                      
 
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->rec->mark_data_ready();
	  }); 
	});
      });
    }
    
    
  };

 

  std::shared_ptr<math_function> define_bad_pixel_correction_function()
  {
    return std::make_shared<cpp_math_function>("snde.bad_pixel_correction",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      executing = make_cppfuncexec_floatingtypes<bad_pixel_correction>(rss,inst);

      if(!executing) {
        executing = make_cppfuncexec_integertypes<bad_pixel_correction>(rss,inst);
        
      }

      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      return executing;      
      
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> bad_pixel_correction_function=define_bad_pixel_correction_function();

  static int registered_bad_pixel_correction_function = register_math_function(bad_pixel_correction_function);
  

    
  };
  
