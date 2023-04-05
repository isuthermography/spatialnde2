#ifndef SNDE_PNGIMAGE_HPP
#define SNDE_PNGIMAGE_HPP

#include <cstdio>


extern "C"
{
  #include <zlib.h>
  #include <png.h>  
}

#include "snde/recstore.hpp"



namespace snde {
  template <typename T>
  void  _store_pngimage_data(std::shared_ptr<ndarray_recording_ref> recref_untyped,png_structp png,png_infop info,png_infop endinfo,size_t width,size_t height)
  {
    //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(output_manager->locker); // new locking process
    // since it's new, we don't have to worry about locking it!

    std::shared_ptr<ndtyped_recording_ref<T>> recref=std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(recref_untyped->assign_recording_type(rtn_typemap.at(typeid(T))));

    recref->allocate_storage({width,height},/*fortran_order=*/true);

    size_t rowcnt;
    png_bytep *row_ptrs = new png_bytep[height];
    for (rowcnt=0;rowcnt < height;rowcnt++) {
      row_ptrs[rowcnt] = png_bytep (recref->shifted_arrayptr()+width*rowcnt);
    }
    png_read_image(png,row_ptrs);
    delete[] row_ptrs;


    png_read_end(png,endinfo);
    
    // set metadata
    png_uint_32 res_x=0,res_y=0;
    int unit_type=0;
    png_get_pHYs(png,info,&res_x,&res_y,&unit_type);
    fprintf(stderr,"res_x=%d; res_y=%d; unit_type=%d\n",res_x,res_y,unit_type);

    std::shared_ptr<constructible_metadata> md=std::make_shared<constructible_metadata>();

    // Reference (0,0) position on image in our coordinates
    // is 0.5 pixel in each axis beyond the bottom left corner
    
    if (unit_type==PNG_RESOLUTION_METER && res_x) {
      md->AddMetaDatum(metadatum("ande_array-axis0_scale",1.0/res_x,"meters"));
      //md->AddMetaDatum(metadatum("IniVal1",-(width*1.0)/res_x/2.0) + 1.0/res_x/2.0);
      md->AddMetaDatum(metadatum("ande_array-axis0_offset",1.0/res_x/2.0,"meters"));
      //md->AddMetaDatum(metadatum("ande_array-axis0_units","meters"));      
    } else {
      md->AddMetaDatum(metadatum("ande_array-axis0_scale",1.0,"pixels"));
      //md->AddMetaDatum(metadatum("IniVal1",-(width*1.0)/2.0)+0.5);
      md->AddMetaDatum(metadatum("ande_array-axis0_offset",0.5,"pixels"));
      //md->AddMetaDatum(metadatum("ande_array-axis0_units","pixels"));      
    }
    md->AddMetaDatum(metadatum("ande_array-axis0_coord","X Position"));

    /* Note for Y axis we put inival positive and step negative so that first pixel 
       in in the upper-left corner, even with our convention  that
       the origin is in the lower-left, 0.5 pixel beyond */
    if (unit_type==PNG_RESOLUTION_METER && res_y) {
      md->AddMetaDatum(metadatum("ande_array-axis1_scale",-1.0/res_y,"meters"));
      //md->AddMetaDatum(metadatum("IniVal2",(height*1.0)/res_y/2.0 -0.5/res_y));
      md->AddMetaDatum(metadatum("ande_array-axis1_offset",(height*1.0)/res_y - 0.5/res_y,"meters"));
      //md->AddMetaDatum(metadatum("ande_array-axis1_units","meters"));
      fprintf(stderr,"Got Y resolution in meters\n");
    } else {
      md->AddMetaDatum(metadatum("ande_array-axis1_scale",-1.0,"pixels"));
      //md->AddMetaDatum(metadatum("IniVal2",(height*1.0)/2.0) - 0.5);
      md->AddMetaDatum(metadatum("ande_array-axis1_offset",(height*1.0) - 0.5,"pixels"));
      //md->AddMetaDatum(metadatum("ande_array-axis1_units","pixels"));      
      fprintf(stderr,"Got Y resolution in arbitrary\n");
    }
    md->AddMetaDatum(metadatum("ande_array-axis1_coord","Y Position"));

    recref->rec->metadata = MergeMetadata(recref->rec->metadata,md);
    
  }

  
  
  static inline void ReadPNG(std::shared_ptr<ndarray_recording_ref> recref_untyped,std::string filename)
  {
    FILE *infile;
    
    png_structp png;
    png_infop info;
    png_infop endinfo;
    short int number = 0x1;
    bool is_little_endian = (bool)*((char*)&number);
    
    png_uint_32 width,height;
    int bit_depth=0, color_type=0,interlace_method=0,compression_method=0,filter_method=0;
    
    infile=fopen(filename.c_str(),"rb");
    if (!infile) return;
    png=png_create_read_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);

    // should png_set_error_fn(...)
    // should add error handling
    
    info=png_create_info_struct(png);
    endinfo=png_create_info_struct(png);
    

    png_init_io(png,infile);
    //png_set_sig_bytes(png,8);

    png_read_info(png,info);

    png_get_IHDR(png,info,&width,&height,&bit_depth,&color_type,&interlace_method,
		 &compression_method,&filter_method);

    if (bit_depth > 8 && is_little_endian) {
      png_set_swap(png);
    }

    if (color_type==PNG_COLOR_TYPE_PALETTE) {
      png_set_palette_to_rgb(png);
    }

    if (color_type==PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
      png_set_expand_gray_1_2_4_to_8(png);
    }
    
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
      png_set_tRNS_to_alpha(png);
    }

    
    if (color_type==PNG_COLOR_TYPE_GRAY_ALPHA || (color_type==PNG_COLOR_TYPE_GRAY && png_get_valid(png,info,PNG_INFO_tRNS))) {
      //png_set_gray_to_rgb(png); // force semitransparent grayscale -> RGBA
      
      png_set_strip_alpha(png); // ignore transparency on grayscale images
    }
    
    if ((color_type==PNG_COLOR_TYPE_RGB || color_type==PNG_COLOR_TYPE_RGBA || color_type==PNG_COLOR_TYPE_PALETTE) && bit_depth > 8) {
      png_set_strip_16(png);
    }
    
    if (color_type==PNG_COLOR_TYPE_RGB || (color_type==PNG_COLOR_TYPE_PALETTE && !png_get_valid(png,info,PNG_INFO_tRNS))) {
      png_set_filler(png,bit_depth > 8 ? 0xffff : 0xff,PNG_FILLER_AFTER);
    }
    
    if (bit_depth < 8) {
      png_set_packing(png);
    }

    // should we do png_set_gamma() here? 

    png_read_update_info(png,info);
    

    
    switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
      if (bit_depth==8) {
	width = png_get_rowbytes(png,info)/sizeof(uint8_t);
	_store_pngimage_data<uint8_t>(recref_untyped,png,info,endinfo,width,height);
      } else if (bit_depth==16) {
	width = png_get_rowbytes(png,info)/sizeof(uint16_t);
	_store_pngimage_data<uint16_t>(recref_untyped,png,info,endinfo,width,height);
	
      } else {
	assert(0); // invalid depth
      }
      break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
    case PNG_COLOR_TYPE_RGB:
    case PNG_COLOR_TYPE_PALETTE:
      
      if (bit_depth==8) {
	width = png_get_rowbytes(png,info)/sizeof(snde_rgba);
	_store_pngimage_data<snde_rgba>(recref_untyped,png,info,endinfo,width,height);
	
      } else {
	assert(0); // invalid depth
      }
      break;

    default:
      assert(0); // bad color_type
    }


    png_destroy_read_struct(&png,&info,&endinfo);
    fclose(infile);
    
  }

}
#endif // SNDE_PNGIMAGE_HPP
