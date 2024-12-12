import spatialnde2 as snde
import sys
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt

specimen_model_file = sys.argv[1]
specimen_model_tree = "/graphics/specimen/"


recdb=snde.recdatabase();
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
recdb.startup()


graphman=snde.graphics_storage_manager("/graphics/",recdb.lowlevel_alloc,recdb.alignment_requirements,recdb.lockmgr,1e-8);

transact = recdb.start_transaction()

graphicsgroup = recdb.define_channel(transact,"/graphics/","main",False,graphman)
specimen_recording = snde.x3d_load_geometry(transact,graphman,specimen_model_file,0,"main",specimen_model_tree,None,[ "reindex_vertices","reindex_tex_vertices","inplanemat","projinfo"])

g=transact.end_transaction().globalrev()


meshedpart = g.get_ndarray_ref("/graphics/specimen/meshed","parts").data
uv = g.get_ndarray_ref("/graphics/specimen/uv","uvs").data
uvcoords2inplane = g.get_ndarray_ref("/graphics/specimen/projinfo","uvcoords2inplane").data

numtris = meshedpart["numtris"][0]
assert(numtris==uv["numuvtris"][0])

# NOTE does not currently attempt to weight by area
dd_dus = np.zeros(numtris,dtype='d')
dd_dvs = np.zeros(numtris,dtype='d')

for trinum in range(numtris):
    #ip2uv = inplane2uvcoords[trinum]["row"][:]["coord"][:,:2] # Ignore offset column
    uv2ip = uvcoords2inplane[trinum]["row"][:]["coord"][:,:2] # Ignore offset column
    # This 2x2 matrix
    # [ a b ] [ u1 ]  -->  in plane coordinates
    # [ c d ] [ v1 ]  -->
    # Thus a movement by some epsilon in u1 moves a distance (eps*sqrt(a^2 + c^2))
    # and a movement by some epsilon in v1 moves a distance (eps*sqrt(b^2 + d^2))
    dd_du = np.sqrt(uv2ip[0,0]**2 + uv2ip[1,0]**2) 
    dd_dv = np.sqrt(uv2ip[0,1]**2 + uv2ip[1,1]**2)

    dd_dus[trinum]=dd_du
    dd_dvs[trinum]=dd_dv
    pass

plt.figure(1)
plt.hist(dd_dus)
plt.title('u axis: median=%g' % (np.median(dd_dus)))

plt.figure(2)
plt.hist(dd_dvs)
plt.title('v axis: median=%g' % (np.median(dd_dvs)))

plt.show()



       
