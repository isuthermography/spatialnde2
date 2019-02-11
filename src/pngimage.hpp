#include <cstdio>


extern "C"
{
  #include <zlib.h>
  #include <png.h>  
}

#include "mutablewfmstore.hpp"

#ifndef SNDE_PNGIMAGE_HPP
#define SNDE_PNGIMAGE_HPP


namespace snde {
  template <typename T>
  std::shared_ptr<mutableelementstore<T>> _store_pngimage_data(std::shared_ptr<arraymanager> manager,std::string Name,png_structp png,png_infop info,png_infop endinfo,size_t width,size_t height)
  {
    //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(output_manager->locker); // new locking process
    // since it's new, we don't have to worry about locking it!

    
    std::shared_ptr<mutableelementstore<T>> retval = std::make_shared<mutableelementstore<T>>(Name,wfmmetadata(),manager,std::vector<snde_index>{width,height},std::vector<snde_index>{1,width});


    size_t rowcnt;
    png_bytep *row_ptrs = new png_bytep[height];
    for (rowcnt=0;rowcnt < height;rowcnt++) {
      row_ptrs[rowcnt] = png_bytep ((*((T **)retval->basearray))+width*rowcnt);
    }
    png_read_image(png,row_ptrs);
    delete[] row_ptrs;


    png_read_end(png,endinfo);
    
    // set metadata
    png_uint_32 res_x=0,res_y=0;
    int unit_type=0;
    png_get_pHYs(png,info,&res_x,&res_y,&unit_type);
    fprintf(stderr,"res_x=%d; res_y=%d; unit_type=%d\n",res_x,res_y,unit_type);

    if (unit_type==PNG_RESOLUTION_METER && res_x) {
      retval->metadata.AddMetaDatum(metadatum("Step1",1.0/res_x));
      retval->metadata.AddMetaDatum(metadatum("IniVal1",-(width*1.0)/res_x/2.0));
      retval->metadata.AddMetaDatum(metadatum("Units1","meters"));      
    } else {
      retval->metadata.AddMetaDatum(metadatum("Step1",1.0));
      retval->metadata.AddMetaDatum(metadatum("IniVal1",-(width*1.0)/2.0));
      retval->metadata.AddMetaDatum(metadatum("Units1","pixels"));      
    }
    retval->metadata.AddMetaDatum(metadatum("Coord1","X Position"));

    /* Note for Y axis we put inival positive and step negative so that first pixel 
       in in the upper-left corner, even with our convention  that
       the origin is in the lower-left */
    if (unit_type==PNG_RESOLUTION_METER && res_y) {
      retval->metadata.AddMetaDatum(metadatum("Step2",-1.0/res_y));
      retval->metadata.AddMetaDatum(metadatum("IniVal2",(height*1.0)/res_y/2.0));
      retval->metadata.AddMetaDatum(metadatum("Units2","meters"));
      fprintf(stderr,"Got Y resolution in meters\n");
    } else {
      retval->metadata.AddMetaDatum(metadatum("Step2",-1.0));
      retval->metadata.AddMetaDatum(metadatum("IniVal2",(height*1.0)/2.0));
      retval->metadata.AddMetaDatum(metadatum("Units2","pixels"));      
      fprintf(stderr,"Got Y resolution in arbitrary\n");
    }
    retval->metadata.AddMetaDatum(metadatum("Coord2","Y Position"));

    manager->mark_as_dirty(nullptr,(void **)retval->basearray,0,width*height);
    
    return retval; 
  }

  
  
  static inline std::shared_ptr<mutabledatastore> ReadPNG(std::shared_ptr<arraymanager> manager,std::string Name,std::string fname)
  // Should probably be called in a transaction in most cases
  // does not add returned datastore to wfmdb -- you have to do that!
  {
    FILE *infile;
    
    unsigned char header[8];
    png_structp png;
    png_infop info;
    png_infop endinfo;
    png_bytep data;
    png_bytep *row_p;
    short int number = 0x1;
    bool is_little_endian = (bool)*((char*)&number);
    std::shared_ptr<mutabledatastore> retval;
    
    double gamma;
    png_uint_32 width,height;
    int bit_depth=0, color_type=0,interlace_method=0,compression_method=0,filter_method=0;
    
    infile=fopen(fname.c_str(),"rb");
    if (!infile) return nullptr;
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
	retval=_store_pngimage_data<uint8_t>(manager,Name,png,info,endinfo,width,height);
      } else if (bit_depth==16) {
	width = png_get_rowbytes(png,info)/sizeof(uint16_t);
	retval=_store_pngimage_data<uint16_t>(manager,Name,png,info,endinfo,width,height);
	
      } else {
	assert(0); // invalid depth
      }
      break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
    case PNG_COLOR_TYPE_RGB:
    case PNG_COLOR_TYPE_PALETTE:
      
      if (bit_depth==8) {
	width = png_get_rowbytes(png,info)/sizeof(snde_rgba);
	retval=_store_pngimage_data<snde_rgba>(manager,Name,png,info,endinfo,width,height);
	
      } else {
	assert(0); // invalid depth
      }
      break;

    default:
      assert(0); // bad color_type
    }


    png_destroy_read_struct(&png,&info,&endinfo);
    fclose(infile);
    
    return retval;
  }

}
#endif // SNDE_PNGIMAGE_HPP
