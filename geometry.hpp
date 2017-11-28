#ifndef SNDE_GEOMETRY_HPP
#define SNDE_GEOMETRY_HPP


/* Thoughts/updates: 
 * For OpenGL compatibility, drop everything down to triangle meshes
 * OpenGL does not support separate UV index. Per malexander in https://www.opengl.org/discussion_boards/showthread.php/183513-How-to-use-UV-Indices-in-OpenGL-4-4  
 * this can be addressed by passing a "Buffer Texture" object to the
 * geometry shader https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader,  which can then (use this to determine texture 
 * coordinates on the actual texture?) (Provide the correct texture
 * index on gl_PrimitiveID? and texture id on a custom output?
 * ... also support 32-bit index storage (snde_shortindex) that gets
 * immediately multiplied out because OpenGL doesn't support 
 * ... 64 bit indices yet. 

*/


#include "geometry_types.h"
#include "geometry.h"

namespace snde {  
  
  /* *** Where to store landmarks ***/
  /* Where to store frames? ***/

  class snde_geometry {
    struct snde_geometrydata geom;

    snde_geometry() {
      
    };
  };
  
}


#endif /* SNDE_GEOMETRY_HPP */
