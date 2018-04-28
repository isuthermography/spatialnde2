import sys
import spatialnde2

# Run this on an X3D file for which the first <Shape> has a texture given by a URL
shapes=spatialnde2.x3d_loader.shapes_from_file(sys.argv[1])

print("Texture URL is \"%s\"" % (shapes[0].nodedata["appearance"].nodedata["texture"].downcast().url))
