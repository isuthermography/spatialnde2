#include "snde/rec_display.hpp"
#include "snde/graphics_recording.hpp"

namespace snde {

  struct chanpathmode_hash {
    size_t operator()(const std::pair<std::string,int>&x) const
    {
      return std::hash<std::string>{}(x.first) + std::hash<int>{}(x.second);
    }
  };
  
  typedef std::unordered_map<std::pair<std::string,int>,std::pair<std::shared_ptr<recording_base>,std::shared_ptr<image_reference>>,chanpathmode_hash> chanpathmode_rectexref_dict;
  
  
static std::string _tdr_tc_join_assem_and_compnames(const std::string &assempath, const std::string compname)
// compname may be relative to our assembly, interpreted as a group
{
  assert(assempath.size() > 0);
  assert(assempath.at(assempath.size()-1) != '/'); // chanpath should not have a trailing '/'
  
  return recdb_path_join(assempath+"/",compname);
}



static bool _tdr_traversetexture(std::shared_ptr<display_info> display,std::shared_ptr<globalrevision> globalrev,chanpathmode_rectexref_dict *channels_modes_imgs,std::shared_ptr<image_reference> texref)
{
  const std::string &chanpath = texref->image_path;
  std::shared_ptr<recording_base> rec = globalrev->get_recording(chanpath);
  
  std::shared_ptr<multi_ndarray_recording> array_rec=std::dynamic_pointer_cast<multi_ndarray_recording>(rec);

  if (array_rec && array_rec->layouts.size() == 1 &&  array_rec->layouts[0].dimlen.size()==texref->other_indices.size()) {
    // if we are a simple ndarray
    channels_modes_imgs->emplace(std::make_pair(chanpath,SNDE_DRM_GEOMETRY),std::make_pair(rec,texref));
    return true;    
  } else {
    return false;
  }
}


  static bool _tdr_traverseparameterization(std::shared_ptr<display_info> display,std::shared_ptr<globalrevision> globalrev, chanpathmode_rectexref_dict *channels_modes_imgs,const std::string & paramname)
{
  std::shared_ptr<recording_base> rec = globalrev->get_recording(paramname);
  

  std::shared_ptr<meshed_parameterization_recording> param_rec=std::dynamic_pointer_cast<meshed_parameterization_recording>(rec);

  
  if (param_rec) {
    // if we are a meshed parameterization recording
    channels_modes_imgs->emplace(std::make_pair(paramname,SNDE_DRM_GEOMETRY),std::make_pair(rec,nullptr));
    return true;    
  } else {
    return false;
  }
}


  static bool _tdr_traversecomponent(std::shared_ptr<display_info> display,std::shared_ptr<globalrevision> globalrev, chanpathmode_rectexref_dict *channels_modes_imgs,const std::string &chanpath)
{
  std::shared_ptr<recording_base> rec = globalrev->get_recording(chanpath);

  std::shared_ptr<assembly_recording> assem_rec=std::dynamic_pointer_cast<assembly_recording>(rec);
  std::shared_ptr<meshed_part_recording> meshed_rec=std::dynamic_pointer_cast<meshed_part_recording>(rec);
  std::shared_ptr<textured_part_recording> texed_rec=std::dynamic_pointer_cast<textured_part_recording>(rec);


  if (assem_rec) {
    channels_modes_imgs->emplace(std::make_pair(chanpath,SNDE_DRM_GEOMETRY),std::make_pair(rec,nullptr));
    for (auto && pathname_orient: assem_rec->pieces) {
      const std::string &raw_pathname = std::get<0>(pathname_orient);
      // raw pathname is relative to our assembly as a group.
      _tdr_traversecomponent(display,globalrev,channels_modes_imgs,_tdr_tc_join_assem_and_compnames(chanpath, raw_pathname));
    }
  
    return true;
  } else if (texed_rec) {
    channels_modes_imgs->emplace(std::make_pair(chanpath,SNDE_DRM_GEOMETRY),std::make_pair(rec,nullptr));

    _tdr_traversecomponent(display,globalrev,channels_modes_imgs,_tdr_tc_join_assem_and_compnames(chanpath, texed_rec->part_name));

    if (texed_rec->parameterization_name) {
      _tdr_traverseparameterization(display,globalrev,channels_modes_imgs,_tdr_tc_join_assem_and_compnames(chanpath, *texed_rec->parameterization_name));
    }

    
    // (***!!! May need to support metadata-based texture overrides here)
    for (auto && facenum_texref: texed_rec->texture_refs) {

      std::string merged_path = _tdr_tc_join_assem_and_compnames(chanpath, facenum_texref.second->image_path);
      std::shared_ptr<image_reference> merged_image_ref=std::make_shared<image_reference>(*facenum_texref.second);
      merged_image_ref->image_path = merged_path;
      _tdr_traversetexture(display,globalrev,channels_modes_imgs,merged_image_ref);
      
    }
    
    return true;
  } else if (meshed_rec) {
    channels_modes_imgs->emplace(std::make_pair(chanpath,SNDE_DRM_GEOMETRY),std::make_pair(rec,nullptr));
    
    return true;
  } else {
    return false;
  }
}


std::vector<display_requirement> traverse_display_requirements(std::shared_ptr<display_info> display,std::shared_ptr<globalrevision> globalrev,const std::vector<std::shared_ptr<display_channel>> &displaychans)
// Assuming the globalrev is fully ready: 
  // Go through the vector of channels we want to display,
  // and figure out
  // (a) all channels that will be necessary, and
  // (b) the math function (if necessary) to render to rgba, and
  // (c) the name of the renderable rgba channel
{
  chanpathmode_rectexref_dict channels_modes_imgs; // set of (channelpath,mode) indexing texture reference pointers (into the recording data structures, but those are immutable and held in memory by the globalrev)
  
  for (auto && displaychan: displaychans) {

    const std::string &chanpath = *displaychan->FullName();
    std::shared_ptr<recording_base> rec = globalrev->get_recording(chanpath);

    /* Figure out type of rendering... */
    int mode=SNDE_DRM_INVALID; 

    std::shared_ptr<multi_ndarray_recording> array_rec=std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
    
    
    if (array_rec && array_rec->layouts.size()==1) {
      // if we are a simple ndarray
      std::shared_ptr<display_axis> axis=display->GetFirstAxis(chanpath);
      
      // Perhaps evaluate/render Max and Min levels here (see scope_drawrec.c)
      snde_index NDim = array_rec->layouts[0].dimlen.size();
      snde_index DimLen1=1;
      if (NDim > 0) {
	DimLen1 = array_rec->layouts[0].dimlen[0];
      }
      
      if (array_rec->layouts[0].flattened_length()==0) {
	continue; // empty array
      }
      
      
      //if (!displaychan->Enabled) {
      //  return nullptr; /* no point in update for a disabled recording */
      //}
      if (NDim<=1 && DimLen1==1) {
	/* "single point" recording */
	fprintf(stderr,"rec_display:traverse_display_requirements(): Single point recording rendering not yet implemented\n");
      } else if (NDim==1) {
	// 1D recording
	fprintf(stderr,"rec_display:traverse_display_requirements(): 1D recording rendering not yet implemented\n");
      } else if (NDim > 1 && NDim <= 4) {
	// image data.. for now hardwired to u=dim 0, v=dim1, frame = dim2, seq=dim3
	std::vector<snde_index> other_indices({0,0});
	if (NDim >= 3) {
	  if (displaychan->DisplayFrame >= array_rec->layouts[0].dimlen[2]) {
	    displaychan->DisplayFrame = array_rec->layouts[0].dimlen[2]-1;	    
	  }
	  other_indices.push_back(displaychan->DisplayFrame);
	  if (NDim >= 4) {
	    if (displaychan->DisplaySeq >= array_rec->layouts[0].dimlen[3]) {
	      displaychan->DisplaySeq = array_rec->layouts[0].dimlen[3]-1;	    
	    }
	    other_indices.push_back(displaychan->DisplaySeq);
	  }
	}
	std::shared_ptr<image_reference> imgref=std::make_shared<image_reference>(chanpath,0,1,other_indices);

	mode = SNDE_DRM_RGBAIMAGE;
	channels_modes_imgs.emplace(std::make_pair(chanpath,mode),std::make_pair(rec,imgref));
	
      }
    } else {
      bool is_geometry = _tdr_traversecomponent(display,globalrev,&channels_modes_imgs,chanpath);
      if (is_geometry) {
	mode = SNDE_DRM_GEOMETRY;
      }
      
    }
    
    if (mode==SNDE_DRM_INVALID) {
      snde_warning("rec_display.cpp: Invalid mode found while traversing display requirements");
    }
  }

  // Gone through all displaychans and accumulated the channels_modes_imgs array.
  // Now create a vector of display_requirement to return
  std::vector<display_requirement> retval;
  for (auto && channel_mode_rec_img: channels_modes_imgs) {
    const std::string &chanpath = channel_mode_rec_img.first.first;
    const int &mode = channel_mode_rec_img.first.second;
    std::shared_ptr<recording_base> &rec = channel_mode_rec_img.second.first;
    std::shared_ptr<image_reference> &imgref = channel_mode_rec_img.second.second;

    std::string renderable_channelpath = chanpath;
    std::shared_ptr<instantiated_math_function> renderable_function;
    // define a transform such as colormapping, etc. if necessary
    // This is also where we'd implement an equivalent to
    // old dataguzzler's ProcRGBA by identifying a transform to
    // run from the recording's metadata
    if (mode==SNDE_DRM_RGBAIMAGE) {
      std::shared_ptr<multi_ndarray_recording> array_rec=std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
      
      if (array_rec && array_rec->layouts.size()==1) {
	// Simple ndarray recording
	std::shared_ptr<ndarray_recording_ref> ref = array_rec->reference_ndarray();
	if (ref->storage->typenum != SNDE_RTN_RGBA32) {
	  // need colormapping */

	  renderable_channelpath = chanpath+"/"+"_snde_rec_colormap";
	  // Will need to do something similar to
	  // recdb->add_math_function() on this
	  // renderable_function to make it run
	  
	  std::shared_ptr<display_channel> displaychan = display->lookup_channel(chanpath);

	  if (displaychan) {
	    renderable_function = display->colormapping_function->instantiate({
		std::make_shared<math_parameter_recording>(chanpath),
		std::make_shared<math_parameter_int_const>(displaychan->ColorMap),
		std::make_shared<math_parameter_double_const>(displaychan->Offset), 
		std::make_shared<math_parameter_double_const>(displaychan->Scale), 
		std::make_shared<math_parameter_indexvec_const>(imgref->other_indices), 
		std::make_shared<math_parameter_int_const>(imgref->u_dimnum), 
		std::make_shared<math_parameter_int_const>(imgref->v_dimnum)
	      },
	      { std::make_shared<std::string>(renderable_channelpath) },
	    "/",
	      false, // is_mutable
	      true, // ondemand
	      false, // mdonly
	      std::make_shared<math_definition>("c++ definition of colormapping"),
	      nullptr); // extra instance parameters -- could have put indexvec, etc. here instead
	  }
	}
      }
    }

    retval.push_back(display_requirement{
	chanpath,
	mode,
	renderable_channelpath,
	renderable_function
      });
    
  }
  
  return retval;
}
};
