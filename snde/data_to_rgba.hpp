#include <mutex>
#include <unordered_map>
#include <cstdint>

#include "snde/geometry_types.h"
#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"
#include "snde/revision_manager.hpp"
#include "snde/mutablerecstore.hpp"
#include "snde/rec_display.hpp"
#include "snde/revman_rec_display.hpp"
#include "snde/revman_recstore.hpp"

#include "snde/geometry_types_h.h"
#include "snde/colormap_h.h"
#include "snde/scale_colormap_c.h"
#include "snde/dummy_scale_colormap_c.h"


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
							      std::shared_ptr<mutablerecdb> recdb,
							      std::string input_fullname,
							      //std::shared_ptr<mutabledatastore> input,
							      //unsigned input_datatype, // MET_...
							      std::shared_ptr<arraymanager> output_manager,
							      void **output_array,
							      std::shared_ptr<display_channel> scaling_colormap_channel,
							      cl_context context,
							      cl_device_id device,
							      cl_command_queue queue,
							      std::function<void(std::shared_ptr<lockholder> input_and_array_locks,rwlock_token_set all_locks,trm_arrayregion input,trm_arrayregion output,snde_rgba **imagearray,snde_index start,size_t xsize,size_t ysize,snde_coord2 inival,snde_coord2 step)> callback, // OK for callback to explicitly unlock locks, as it is the last thing called.
							      std::function<void(void)> cleanup)
  {
    
    std::shared_ptr<trm_dependency> retval;
    std::vector<trm_struct_depend> struct_inputs;

    struct_inputs.emplace_back(display_channel_dependency(revman,scaling_colormap_channel));
    struct_inputs.emplace_back(rec_dependency(revman,recdb,input_fullname));
    
    retval=revman->add_dependency_during_update(						
						struct_inputs,
						std::vector<trm_arrayregion>(), // inputs
						std::vector<trm_struct_depend>(), // struct_outputs
						// function
						[ context,device,queue,output_array,output_manager,callback ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {
						  std::shared_ptr<display_channel> scaling_colormap_channel=get_display_channel_dependency(dep->struct_inputs[0]);
						  std::shared_ptr<mutablerecdb> recdb;
						  std::shared_ptr<mutableinfostore> rec_inp;
						  std::shared_ptr<mutabledatastore> rec_inp_data;
						  
						  std::tie(recdb,rec_inp) = get_rec_dependency(dep->struct_inputs[1]);
						  
						  
						  if (recdb && rec_inp) {
						    rec_inp_data = std::dynamic_pointer_cast<mutabledatastore>(rec_inp);
						  }
						  
						  if (!recdb || !rec_inp || !rec_inp_data || !scaling_colormap_channel) {
						    // data invalid or not available: inputs & outputs blank
						    std::vector<trm_arrayregion> new_inputs;
						    dep->update_inputs(new_inputs);
						    if (actions & STDA_IDENTIFYOUTPUTS) {
						      std::vector<trm_arrayregion> new_outputs;
						      dep->update_outputs(new_outputs);
						    }
						    
						    return;
						  }
						  
						  
						  
						  // perform locking
						  
						  // obtain lock for input structure (prior to all arrays in locking order)
						  
						  std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						  std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(output_manager->locker); // new locking process
						  
						  holder->store(lockprocess->get_locks_read_lockable(rec_inp));
						  
						  // Use spawn to get the array locks, as we don't know relative positions of input and output arrays in the locking roder
						  lockprocess->spawn( [ dep,rec_inp_data,lockprocess,holder ]() {
									
									holder->store(lockprocess->get_locks_read_array_region(rec_inp_data->basearray,rec_inp_data->startelement,rec_inp_data->numelements));
								      });
						  
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    lockprocess->spawn( [ rec_inp_data,output_array,output_manager,dep,lockprocess,holder ]() { holder->store_alloc(dep->realloc_output_if_needed(lockprocess,output_manager,0,output_array,rec_inp_data->numelements,"output"));});
						    
						  }
						  rwlock_token_set all_locks=lockprocess->finish();
						  
						
						  std::vector<trm_arrayregion> new_inputs;
						
						  new_inputs.push_back(trm_arrayregion(rec_inp_data->manager,rec_inp_data->basearray,rec_inp_data->startelement,rec_inp_data->numelements));
						  
						  dep->update_inputs(new_inputs);
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    std::vector<trm_arrayregion> new_outputs;
						    dep->add_output_to_array(new_outputs,output_manager,holder,0,output_array,"output");
						    dep->update_outputs(new_outputs);
						    
						    if (actions & STDA_EXECUTE) {
						      // function code
						      float Offset;
						      float alpha_float;
						      float DivPerUnits;
						      uint8_t Alpha;
						      size_t DisplayFrame;
						      size_t DisplaySeq;
						      snde_index ColorMap;
						      
						      // extract mutableinfostore, which should come from dep->struct_inputs.at(1).first.keyimpl which
						      
						      
						      snde_coord2 inival={
									  rec_inp_data->metadata.GetMetaDatumDbl("IniVal1",0.0),
									  rec_inp_data->metadata.GetMetaDatumDbl("IniVal2",0.0),
						      };
						      
						      snde_coord2 step={
									rec_inp_data->metadata.GetMetaDatumDbl("Step1",1.0),
									rec_inp_data->metadata.GetMetaDatumDbl("Step2",1.0),
						      };
						      
						      
						      cl_kernel scale_colormap_kern;
						      // obtain kernel
						      {
							std::lock_guard<std::mutex> scop_lock(scop_mutex);
							auto scop_iter = scale_colormap_opencl_program.find(rec_inp_data->typenum);
							if (scop_iter==scale_colormap_opencl_program.end()) {
							
							  
							  scale_colormap_opencl_program.emplace(std::piecewise_construct,std::forward_as_tuple(rec_inp_data->typenum),std::forward_as_tuple(std::string("scale_colormap"), std::vector<std::string>{ get_data_to_rgba_program_text(rec_inp_data->typenum) }));
							}
							
							scale_colormap_kern = scale_colormap_opencl_program.at(rec_inp_data->typenum).get_kernel(context,device);
							
						      }
						      
						      // extract parameters from scaling_colormap_channel, which should come from dep->struct_inputs.at(0)
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
						      
						      
						      
						      // Now transfer from input to output while colormapping, scaling,etc.	
						      size_t xaxis=0;
						      size_t yaxis=1;
						      size_t frameaxis=2;
						      size_t seqaxis=3;
						      
						      snde_index xsize=0;
						      snde_index ysize=0;
						      snde_index xstride=0;
						      snde_index ystride=0;
						      if (rec_inp_data->dimlen.size() >= xaxis+1) {
							xsize=rec_inp_data->dimlen[xaxis];
							xstride=rec_inp_data->strides[xaxis];
						      }
						      if (rec_inp_data->dimlen.size() >= yaxis+1) {
							ysize=rec_inp_data->dimlen[yaxis];
							ystride=rec_inp_data->strides[yaxis];
						      }
						      if (xsize*ysize > dep->outputs[0].len) {
							ysize=dep->outputs[0].len/xsize; // just in case output is somehow too small (might happen in loose consistency mode)
						      }
						    
						      if (frameaxis >= rec_inp_data->dimlen.size()) {
							DisplayFrame=0;
						      } else if (DisplayFrame >= rec_inp_data->dimlen[frameaxis]) {
							DisplayFrame=rec_inp_data->dimlen[frameaxis]-1;
						      }
						      
						      if (seqaxis >= rec_inp_data->dimlen.size()) {
							DisplaySeq=0;
						      } else if (DisplaySeq >= rec_inp_data->dimlen[seqaxis]) {
							DisplaySeq=rec_inp_data->dimlen[seqaxis]-1;
						      }
						      
						      //std::vector<snde_index> dimlen=rec_inp_data->dimlen;
						      //std::vector<snde_index> strides=rec_inp_data->strides;
						      //snde_index startelement=rec_inp_data->startelement;
						      snde_index input_offset = 0;
						      size_t axcnt;
						      for (axcnt=0;axcnt < rec_inp_data->dimlen.size();axcnt++) {
							if (axcnt==frameaxis) {
							  input_offset += DisplayFrame*rec_inp_data->strides[axcnt];
							}
							if (axcnt==seqaxis) {
							  input_offset += DisplaySeq*rec_inp_data->strides[axcnt];
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
						      
						      
						      
						      // Release our reference to kernel, allowing it to be free'd
						      clReleaseKernel(scale_colormap_kern);
						      
						      // while we still have our locks (all_locks),
						      // call callback function with the data we have generated 
						      // OK for callback to explicitly unlock locks
						      callback(holder,std::move(all_locks),dep->inputs[0],dep->outputs[0],(snde_rgba **)dep->outputs[0].array,dep->outputs[0].start,xsize,ysize,inival,step);
						      
						      // cannot use inputlock anymore after this because of std::move... (of course we are done anyway)
						      
						    }
						    
						  }
						},
						[ cleanup ] (trm_dependency *dep)  {
						  // cleanup function
						  cleanup();

						  // free our outputs
						  std::vector<trm_arrayregion> new_outputs;
						  dep->free_output(new_outputs,0);
						  dep->update_outputs(new_outputs);
						  
						});
    
    return retval;
    
  }
  
  
};

#endif // SNDE_DATA_TO_RGBA_HPP
