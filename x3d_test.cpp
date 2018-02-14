
#include "x3d.hpp"


int main(int argc, char **argv)
{
  assert(argc==2);
  std::vector<std::shared_ptr<snde::x3d_shape>> shapes=snde::x3d_loader::shapes_from_file(argv[1]);
  std::shared_ptr<snde::x3d_indexedfaceset> first_shape=std::dynamic_pointer_cast<snde::x3d_indexedfaceset>(shapes[0]->nodedata["geometry"]);

  return 0;
}
