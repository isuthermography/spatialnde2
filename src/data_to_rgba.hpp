#include <mutex>
#include <unordered_map>
#include <cstdint>

#include "geometry_types.h"
#include "openclcachemanager.hpp"
#include "opencl_utils.hpp"
#include "revision_manager.hpp"
#include "mutablewfmstore.hpp"
#include "wfm_display.hpp"
#include "revman_wfm_display.hpp"
#include "revman_wfmstore.hpp"

#include "geometry_types_h.h"
#include "colormap_h.h"
#include "scale_colormap_c.h"
#include "dummy_scale_colormap_c.h"


#ifndef SNDE_DATA_TO_RGBA_HPP
#define SNDE_DATA_TO_RGBA_HPP

namespace snde {

  extern std::mutex scop_mutex; // for scale_colormap_opencl_program
  extern std::unordered_map<unsigned,opencl_program> scale_colormap_opencl_program; // indexed by input_datatype (MET_...); locked by scop_mutex;



static inline std::string get_data_to_rgba_program_text(unsigned input_datatype)
  {
    // return the code to use for data->rgba conversion,
    std::string maincode;
    if (input_datatype==MET_RGBA32) {
      maincode = dummy_scale_colormap_c;
    } else {
      maincode = scale_colormap_c;
    }
    
    return std::string(geometry_types_h) + colormap_h + "\ntypedef " + met_ocltypemap.at(input_datatype) + " sc_intype;\n" + maincode;

    
  };
  
  /* *** NOTE: CreateTextureDependency should be called during a revman transaction and 
     will lock arrays */
  /* ***!!!! NEED A WAY TO OBTAIN A READ LOCK ON THE DEPENDENCY OUTPUT TO HOLD WHILE RENDERING!!!*** */
  /* (Idea: just lock vertex_arrays, texvertex_arrays, and texbuffer) */
  /* ***!!! Should have ability to combine texture data from multiple patches... see geometry_types.h */ 
  static std::shared_ptr<trm_dependency> CreateRGBADependency(std::shared_ptr<trm> revman,
							      std::shared_ptr<mutablewfmdb> wfmdb,
							      std::string input_fullname,
							      //std::shared_ptr<mutabledatastore> input,
							      //unsigned input_datatype, // MET_...
							      std::shared_ptr<arraymanager> output_manager,
							      void **output_array,
							      std::shared_ptr<display_channel> scaling_colormap_channel,
							      cl_context context,
							      cl_device_id device,
							      cl_command_queue queue,
							      std::function<void(std::shared_ptr<lockholder> input_and_array_locks,rwlock_token_set all_locks,trm_arrayregion input,trm_arrayregion output,snde_rgba **imagearray,snde_index start,size_t xsize,size_t ysize,snde_coord2 inival,snde_coord2 step)> callback) // OK for callback to explicitly unlock locks, as it is the last thing called. 
  {
    
    std::shared_ptr<trm_dependency> retval;
    std::vector<trm_struct_depend> struct_inputs;

    struct_inputs.emplace_back(display_channel_dependency(revman,scaling_colormap_channel));
    struct_inputs.emplace_back(wfm_dependency(revman,wfmdb,input_fullname));
    
    retval=revman->add_dependency_during_update([ context, device, queue, output_manager, output_array, callback ] (snde_index newversion,std::shared_ptr<trm_dependency> dep, std::vector<rangetracker<markedregion>> &inputchangedregions) {
	// function code
	float Offset;
	float alpha_float;
	float DivPerUnits;
	uint8_t Alpha;
	size_t DisplayFrame;
	size_t DisplaySeq;
	snde_index ColorMap;

	// extract mutableinfostore, which should come from dep->struct_inputs.at(1).first.keyimpl which
	// is a trm_mutablewfm_key that has wfmdb and wfmfullname members
	std::shared_ptr<mutablewfmdb> wfmdb = std::dynamic_pointer_cast<trm_mutablewfm_key>(dep->struct_inputs.at(1).first.keyimpl)->wfmdb.lock();
	std::shared_ptr<mutabledatastore> input=std::dynamic_pointer_cast<mutabledatastore>(wfmdb->lookup(std::dynamic_pointer_cast<trm_mutablewfm_key>(dep->struct_inputs.at(1).first.keyimpl)->wfmfullname));
	

	snde_coord2 inival={
			    input->metadata.GetMetaDatumDbl("IniVal1",0.0),
			    input->metadata.GetMetaDatumDbl("IniVal2",0.0),
	};

	snde_coord2 step={
			  input->metadata.GetMetaDatumDbl("Step1",1.0),
			  input->metadata.GetMetaDatumDbl("Step2",1.0),
	};

	
	cl_kernel scale_colormap_kern;
	// obtain kernel
	{
	  std::lock_guard<std::mutex> scop_lock(scop_mutex);
	  auto scop_iter = scale_colormap_opencl_program.find(input->typenum);
	  if (scop_iter==scale_colormap_opencl_program.end()) {

	    
	    scale_colormap_opencl_program.emplace(std::piecewise_construct,std::forward_as_tuple(input->typenum),std::forward_as_tuple(std::string("scale_colormap"), std::vector<std::string>{ get_data_to_rgba_program_text(input->typenum) }));
	  }
	  
	  scale_colormap_kern = scale_colormap_opencl_program.at(input->typenum).get_kernel(context,device);
	  
	}

	// extract parameters from scaling_colormap_channel, which should come from dep->struct_inputs.at(0)
	std::shared_ptr<display_channel> scaling_colormap_channel = std::dynamic_pointer_cast<trm_wfmdisplay_key>(dep->struct_inputs.at(0).first.keyimpl)->displaychan;//.lock();
	
	{
	  std::lock_guard<std::mutex> displaychan_lock(scaling_colormap_channel->admin);
	  Offset=scaling_colormap_channel->Offset;
	  alpha_float=roundf(scaling_colormap_channel->Alpha*255.0);
	  if (alpha_float < 0.0) alpha_float=0.0;
	  if (alpha_float > 255.0) alpha_float=255.0;
	  Alpha=alpha_float;
	  DisplayFrame=scaling_colormap_channel->DisplayFrame;
	  DisplaySeq=scaling_colormap_channel->DisplaySeq;
	  ColorMap=scaling_colormap_channel->ColorMap;
	  DivPerUnits = 1.0/scaling_colormap_channel->Scale; // !!!*** Should we consider pixelflag here? Probably not because color axis can't be in pixels, so it wouldn't make sense
	}
	
	
	// perform locking

	// obtain lock for input structure (prior to all arrays in locking order)
	//std::unique_lock<rwlock_lockable> inputlock(input->lock->reader);

	//// obtain lock for output structure (prior to all arrays in locking order)
	//std::unique_lock<rwlock_lockable> outputlock(output->lock->writer);
	
	std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
	std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(output_manager->locker); // new locking process

	// Use spawn to get the locks, as we don't know where we are in the locking order
	lockprocess->spawn( [ dep,input,lockprocess,holder ]() {
			      holder->store(lockprocess->get_locks_read_lockable(input));
			      holder->store(lockprocess->get_locks_read_array_region(dep->inputs[0].array,dep->inputs[0].start,dep->inputs[0].len));
			    });
	lockprocess->spawn( [ output_array,dep,lockprocess,holder ]() { holder->store(lockprocess->get_locks_write_array_region(output_array,dep->outputs[0].start,dep->outputs[0].len)); });
	
	rwlock_token_set all_locks=lockprocess->finish();

	// Now transfer from input to output while colormapping, scaling,etc.	
	size_t xaxis=0;
	size_t yaxis=1;
	size_t frameaxis=2;
	size_t seqaxis=3;

	snde_index xsize=0;
	snde_index ysize=0;
	snde_index xstride=0;
	snde_index ystride=0;
	if (input->dimlen.size() >= xaxis+1) {
	  xsize=input->dimlen[xaxis];
	  xstride=input->strides[xaxis];
	}
	if (input->dimlen.size() >= yaxis+1) {
	  ysize=input->dimlen[yaxis];
	  ystride=input->strides[yaxis];
	}
	if (xsize*ysize > dep->outputs[0].len) {
	  ysize=dep->outputs[0].len/xsize; // just in case output is somehow too small (might happen in loose consistency mode)
	}

	if (frameaxis >= input->dimlen.size()) {
	  DisplayFrame=0;
	} else if (DisplayFrame >= input->dimlen[frameaxis]) {
	  DisplayFrame=input->dimlen[frameaxis]-1;
	}

	if (seqaxis >= input->dimlen.size()) {
	  DisplaySeq=0;
	} else if (DisplaySeq >= input->dimlen[seqaxis]) {
	  DisplaySeq=input->dimlen[seqaxis]-1;
	}
	
	//std::vector<snde_index> dimlen=input->dimlen;
	//std::vector<snde_index> strides=input->strides;
	//snde_index startelement=input->startelement;
	snde_index input_offset = 0;
	size_t axcnt;
	for (axcnt=0;axcnt < input->dimlen.size();axcnt++) {
	  if (axcnt==frameaxis) {
	    input_offset += DisplayFrame*input->strides[axcnt];
	  }
	  if (axcnt==seqaxis) {
	    input_offset += DisplaySeq*input->strides[axcnt];
	  }	  
	}
	if (input_offset > dep->inputs[0].len) {
	  xsize=0;
	  ysize=0;
	  input_offset=0;
	} else {
	  if (input_offset + xstride*xsize + ystride*ysize > dep->inputs[0].len) {
	    if (ystride*ysize > xstride*xsize) {
	      ysize = (dep->inputs[0].len - input_offset - xstride*xsize)/ystride;
	    } else {
	      xsize = (dep->inputs[0].len - input_offset - ystride*ysize)/xstride;

	    }
	  }
	}
	assert(input_offset + xstride*xsize + ystride*ysize <= dep->inputs[0].len);
	
	
	OpenCLBuffers Buffers(context,device,all_locks);
	Buffers.AddSubBufferAsKernelArg(dep->inputs[0].manager,scale_colormap_kern,0,dep->inputs[0].array,dep->inputs[0].start,dep->inputs[0].len,false);
	Buffers.AddSubBufferAsKernelArg(dep->outputs[0].manager,scale_colormap_kern,1,dep->outputs[0].array,dep->outputs[0].start,dep->outputs[0].len,true);
	//snde_index strides[2]={xstride,ystride};
	clSetKernelArg(scale_colormap_kern,2,sizeof(input_offset),&input_offset);
	clSetKernelArg(scale_colormap_kern,3,sizeof(xstride),&xstride);
	clSetKernelArg(scale_colormap_kern,4,sizeof(ystride),&ystride);
	clSetKernelArg(scale_colormap_kern,5,sizeof(Offset),&Offset);
	clSetKernelArg(scale_colormap_kern,6,sizeof(Alpha),&Alpha);
	clSetKernelArg(scale_colormap_kern,7,sizeof(ColorMap),&ColorMap);
	clSetKernelArg(scale_colormap_kern,8,sizeof(DivPerUnits),&DivPerUnits);

						    
	size_t worksize[2]={xsize,ysize};
	cl_event kernel_complete=NULL;
	
	fprintf(stderr,"Converting data to RGBA\n");
	
	cl_int err=clEnqueueNDRangeKernel(queue,scale_colormap_kern,2,NULL,worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);

	if (err != CL_SUCCESS) {
	  throw openclerror(err,"Error enqueueing kernel");
	}
	
	clFlush(queue); /* trigger execution */

	Buffers.SubBufferDirty(dep->outputs[0].array,dep->outputs[0].start,dep->outputs[0].len);
	Buffers.RemBuffers(kernel_complete,kernel_complete,true); /* wait for completion */
	clReleaseEvent(kernel_complete);


	/*	{
	  FILE *testout;
	  testout=fopen("/tmp/testout.bin","wb");
	  fwrite((*((snde_index **)dep->outputs[0].array))+dep->outputs[0].start,4,xsize*ysize,testout);
	  fclose(testout);
	  }*/
	
	// Release our reference to kernel, allowing it to be free'd
	clReleaseKernel(scale_colormap_kern);

	// while we still have our locks (all_locks),
	// call callback function with the data we have generated 
	// OK for callback to explicitly unlock locks
	callback(holder,std::move(all_locks),dep->inputs[0],dep->outputs[0],(snde_rgba **)dep->outputs[0].array,dep->outputs[0].start,xsize,ysize,inival,step);

	// cannot use inputlock anymore after this because of std::move... (of course we are done anyway)
	
      },
      [  ] (std::vector<trm_struct_depend> struct_inputs,std::vector<trm_arrayregion> inputs) -> std::vector<trm_arrayregion> {
	/* regionupdater code */
	if (inputs.size() > 0) {
	  inputs.empty();
	}

	// extract mutableinfostore, which should come from dep->struct_inputs.at(1).first.keyimpl which
	// is a trm_mutablewfm_key that has wfmdb and wfmfullname members
	std::shared_ptr<mutablewfmdb> wfmdb = std::dynamic_pointer_cast<trm_mutablewfm_key>(struct_inputs.at(1).first.keyimpl)->wfmdb.lock();
	std::shared_ptr<mutabledatastore> input=std::dynamic_pointer_cast<mutabledatastore>(wfmdb->lookup(std::dynamic_pointer_cast<trm_mutablewfm_key>(struct_inputs.at(1).first.keyimpl)->wfmfullname));

	inputs.push_back(trm_arrayregion(input->manager,input->basearray,input->startelement,input->numelements));
	return inputs; 
      },
      struct_inputs, // struct_inputs
      std::vector<trm_arrayregion>(), // inputs
      std::vector<trm_struct_depend>(), // struct_outputs
      [ output_manager,output_array ] (std::vector<trm_struct_depend> struct_inputs, std::vector<trm_arrayregion> inputs,std::vector<trm_struct_depend> struct_outputs,std::vector<trm_arrayregion> outputs) -> std::vector<trm_arrayregion> {
	/* update_output_regions code */

	// extract mutableinfostore, which should come from dep->struct_inputs.at(1).first.keyimpl which
	// is a trm_mutablewfm_key that has wfmdb and wfmfullname members
	std::shared_ptr<mutablewfmdb> wfmdb = std::dynamic_pointer_cast<trm_mutablewfm_key>(struct_inputs.at(1).first.keyimpl)->wfmdb.lock();
	std::shared_ptr<mutabledatastore> input=std::dynamic_pointer_cast<mutabledatastore>(wfmdb->lookup(std::dynamic_pointer_cast<trm_mutablewfm_key>(struct_inputs.at(1).first.keyimpl)->wfmfullname));
	
	// obtain lock for input structure (prior to all arrays in locking order
	rwlock_token_set all_locks=empty_rwlock_token_set();
	input->manager->locker->get_locks_read_lockable(all_locks,input);
	
	//std::lock_guard<rwlock_lockable> inputlock(input->lock->reader);
	std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
	snde_index input_len = 0;
	if (input->dimlen.size() >= 1) {
	  input_len=input->dimlen[0];
	}
	if (input->dimlen.size() >= 2) {
	  input_len*=input->dimlen[1];
	}
	
	if (outputs.size() >= 1 && outputs[0].len != input_len) {
	  if (outputs[0].start != SNDE_INDEX_INVALID) {
	    output_manager->free(output_array,outputs[0].start);
	  }
	  outputs.empty();
	}
	if (outputs.size() < 1) {
	  std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(output_manager->locker); // new locking process
	  
	  lockprocess->get_locks_write_array(output_array);
	  
	  rwlock_token_set all_locks=lockprocess->finish();

	  snde_index start;
	  std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> allocation_vector;
	  std::tie(start,allocation_vector)=output_manager->alloc_arraylocked(all_locks,output_array,input_len);
	  
	  outputs.push_back(trm_arrayregion(output_manager,output_array,start,input_len));
	  

	  return outputs;
	  
	}
	return outputs;
      } ,
      [ output_manager,output_array ](std::vector<trm_struct_depend> struct_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs) {
	// cleanup
	if (outputs.size()==1) {
	  output_manager->free(output_array,outputs[0].start);
	}
      });
  
    // mark that our dep must be rerun if the channel params change. (NOW USING revman_wfm_display.hpp) via the call to display_channel_dependency()
    
    //scaling_colormap_channel->adjustment_deps.emplace(std::weak_ptr<trm_dependency>(retval));
    return retval;
      
  }
  
  
};

#endif // SNDE_DATA_TO_RGBA_HPP
