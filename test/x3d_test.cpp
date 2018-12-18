#include <cstdio>

#include "x3d.hpp"


int main(int argc,char **argv)
{
  assert(argc==2);
  // Run this on an X3D file for which the first <Shape> has an indexedfaceset geometry and a  texture given by a URL
  std::vector<std::shared_ptr<snde::x3d_shape>> shapes=snde::x3d_loader::shapes_from_file(argv[1]);
  std::shared_ptr<snde::x3d_indexedfaceset> first_ifs=std::dynamic_pointer_cast<snde::x3d_indexedfaceset>(shapes[0]->nodedata["geometry"]);

  if (shapes[0]->nodedata["appearance"]->nodedata["texture"]) {
    printf("Texture URL is \"%s\"\n",std::dynamic_pointer_cast<snde::x3d_imagetexture>(shapes[0]->nodedata["appearance"]->nodedata["texture"])->url.c_str());
  }
  
  return 0;
}
