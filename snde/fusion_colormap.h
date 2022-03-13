#ifndef SNDE_FUSION_COLORMAP_H
#define SNDE_FUSION_COLORMAP_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef snde_float32 fc_real_intype;
  typedef snde_complexfloat32 fc_complex_intype;
  
  snde_rgba snde_fusion_colormap_real(const fc_real_intype inval,
				      const snde_float32 total,
				      uint32_t colormap_type,
				      snde_float32 offset,
				      snde_float32 unitsperintensity,
				      snde_float32 maxtotal,
				      uint8_t alpha);
  
  
  snde_rgba snde_fusion_colormap_complex(const fc_complex_intype inval,
					 const snde_float32 total,
					 uint32_t colormap_type,
					 snde_float32 offset,
					 snde_float32 unitsperintensity,
					 snde_float32 maxtotal,
					 uint8_t alpha);



#ifdef __cplusplus
};
#endif


#endif // SNDE_FUSION_COLORMAP_H
