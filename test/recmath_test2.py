import sys
import os

import multiprocessing 
import math


import spatialnde2 as snde
from spatialnde2_example_external_cpp_function import scalar_multiply_function

rec_len=100
scalefactor=4.5

recdb=snde.recdatabase()

recdb.default_storage_manager = snde.recording_storage_manager_shmem()

cpu_compute = snde.available_compute_resource_cpu(recdb,recdb.compute_resources,snde.SNDE_CR_CPU,multiprocessing.cpu_count())
recdb.compute_resources.compute_resources.append(cpu_compute) #std::thread::hardware_concurrency()))
recdb.compute_resources.start()


scaled_channel_function = scalar_multiply_function.instantiate([ snde.math_parameter_recording("/test_channel"), snde.math_parameter_double_const(scalefactor) ],
                                                               [ snde.shared_string("scaled_channel") ],
                                                               "/",
                                                               False,
                                                               False,
                                                               False,
                                                               snde.math_definition("c++ definition"),
                                                               None)
transact = snde.active_transaction(recdb); # Transaction RAII holder

recdb.add_math_function(scaled_channel_function,False)

testchan_config=snde.channelconfig("/test_channel", "main", recdb,False)
  
testchan = recdb.reserve_channel(testchan_config);

# demonstrate alternative ways to create the recording

test_rec_32 = snde.ndarray_recording.create_typed_recording(recdb,testchan,recdb,snde.SNDE_RTN_FLOAT32)

globalrev = transact.end_transaction()

transact2 = snde.active_transaction(recdb); # Transaction RAII holder
test_rec_64 = snde.ndarray_recording.create_typed_recording(recdb,testchan,recdb,snde.SNDE_RTN_FLOAT64)
globalrev2 = transact2.end_transaction()


test_rec_32.metadata=snde.immutable_metadata()
test_rec_32.mark_metadata_done()
test_rec_32.allocate_storage([ rec_len ]);

test_rec_64.metadata=snde.immutable_metadata()
test_rec_64.mark_metadata_done()
test_rec_64.allocate_storage([ rec_len ]);

for cnt in range(rec_len):
    test_rec_32.assign_double([cnt],100.0*math.sin(cnt))
    test_rec_64.assign_double([cnt],100.0*math.sin(cnt))
    pass

test_rec_32.mark_as_ready()
test_rec_64.mark_as_ready()

globalrev.wait_complete();
globalrev2.wait_complete();

scaled_rec_32 = globalrev.get_recording("/scaled_channel").cast_to_ndarray()
data_32 = scaled_rec_32.data()

for cnt in range(rec_len):
    math_function_value = data_32[cnt]
    recalc_value = test_rec_32.data()[cnt]*scalefactor
    print(" %f \t \t %f" % (recalc_value,math_function_value)) 
    assert(abs(math_function_value-recalc_value) < 1e-4) # No functionality in Python to do single precision calculation for comparison
    pass

scaled_rec_64 = globalrev2.get_recording("/scaled_channel").cast_to_ndarray()
data_64 = scaled_rec_64.data()

for cnt in range(rec_len):
    math_function_value = data_64[cnt]
    recalc_value = test_rec_64.data()[cnt]*scalefactor
    print(" %f \t \t %f" % (recalc_value,math_function_value)) 
    assert(math_function_value == recalc_value) 
    pass
