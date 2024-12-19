import sys
import os

import multiprocessing 
import math

import numpy as np

import spatialnde2 as snde

recdb=snde.recdatabase();
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
snde.setup_opencl(recdb,[],False,8,None) # limit to 8 parallel jobs. Could replace nullptr with OpenCL platform nameo
recdb.startup()

kdtree_math_function = recdb.lookup_available_math_function("snde.kdtree_calculation")

kdtree_function = kdtree_math_function.instantiate([ snde.math_parameter_recording("/match_vertices"),  ],
                                                       [ snde.shared_string("/kdtree") ],
                                                       "/",
                                                       False,
                                                       False,
                                                       False,
                                                       snde.math_definition("kdtree definition"),
                                                   [],
                                                       None)

knn_math_function = recdb.lookup_available_math_function("snde.knn_calculation")

knn_function = knn_math_function.instantiate([ snde.math_parameter_recording("/match_vertices"),
                                               snde.math_parameter_recording("/kdtree"),
                                               snde.math_parameter_recording("/search_points"),
                                              ],
                                             [ snde.shared_string("/knn") ],
                                             "/",
                                             False,
                                             False,
                                             False,
                                             snde.math_definition("knn definition"),
                                             [],
                                             None)


transact = recdb.start_transaction(); # Transaction RAII holder

recdb.add_math_function(transact,kdtree_function,False)
recdb.add_math_function(transact,knn_function,False)

num_raw_vertices = 2000
#num_raw_vertices = 13
vertices_config=snde.channelconfig("/match_vertices", "main",False)
vertices_chan = recdb.reserve_channel(transact,vertices_config);
vertices = snde.create_ndarray_ref(transact,vertices_chan,snde.SNDE_RTN_SNDE_COORD3)
vertices.rec.metadata=snde.immutable_metadata()
vertices.rec.mark_metadata_done()
vertices.allocate_storage([ num_raw_vertices ]);
raw_vertices = np.random.randn(num_raw_vertices,3)

vertices.data["coord"] = raw_vertices
vertices.rec.mark_data_ready()



num_search_points = 10000
#num_search_points = 1
search_points_config=snde.channelconfig("/search_points", "main",False)
search_points_chan = recdb.reserve_channel(transact,search_points_config);
search_points = snde.create_ndarray_ref(transact,search_points_chan,snde.SNDE_RTN_SNDE_COORD3)
search_points.rec.metadata=snde.immutable_metadata()
search_points.rec.mark_metadata_done()
search_points.allocate_storage([ num_search_points ]);
#raw_search_points = np.array([ 
#    (4,8, 0 ),
#],dtype=np.float32)
np.random.seed(0)
raw_search_points = np.random.randn(num_search_points,3)*8
search_points.data["coord"] = raw_search_points
search_points.rec.mark_data_ready()


    
globalrev = transact.end_transaction().globalrev()


kdtree = globalrev.get_ndarray_ref("/kdtree","vertex_kdtree").data
knn = globalrev.get_ndarray_ref("/knn").data


by_numpy = np.argmin(np.linalg.norm((raw_vertices[np.newaxis,:,:]-raw_search_points[:,np.newaxis,:]),axis=2),axis=1)

assert((knn==by_numpy).all()) # Note: infintesimal possibility of a transient failure here if a point in the search points is exactly equidistant (or within machine precision) to multiple vertices. 

# Do a second transaction where we feed in the vertices as the search points
# and verify that case works as well

transact2 = recdb.start_transaction(); # Transaction RAII holder
search_points2 = snde.create_ndarray_ref(transact2,search_points_chan,snde.SNDE_RTN_SNDE_COORD3)
search_points2.rec.metadata=snde.immutable_metadata()
search_points2.rec.mark_metadata_done()
search_points2.allocate_storage([ num_raw_vertices ]);
#raw_search_points = np.array([ 
#    (4,8, 0 ),
#],dtype=np.float32)
search_points2.data["coord"] = raw_vertices
search_points2.rec.mark_data_ready()
globalrev2 = transact2.end_transaction().globalrev()



knn2 = globalrev2.get_ndarray_ref("/knn").data


by_numpy2 = np.argmin(np.linalg.norm((raw_vertices[np.newaxis,:,:]-raw_vertices[:,np.newaxis,:]),axis=2),axis=1)

assert((knn2==by_numpy2).all()) # Note: infintesimal possibility of a transient failure here if a vertex is doubles. 
