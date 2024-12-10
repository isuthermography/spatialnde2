import sys
import os

import multiprocessing 
import math


import spatialnde2 as snde
from spatialnde2_example_external_cpp_function import scalar_multiply_function

rec_len=100
scalefactor=4.5

recdb=snde.recdatabase();
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
recdb.startup()


scaled_channel_function = scalar_multiply_function.instantiate([ snde.math_parameter_recording("/test_channel"), snde.math_parameter_double_const(scalefactor) ],
                                                               [ snde.shared_string("scaled_channel") ], # Note that this is relative to the context given below, which is "/"
                                                               "/",
                                                               False,
                                                               False,
                                                               False,
                                                               snde.math_definition("c++ definition"),
                                                               [],
                                                               None)
transact = recdb.start_transaction(); # Transaction RAII holder

recdb.add_math_function(transact,scaled_channel_function,False)

#scaled_channel_function2 = scalar_multiply_function.instantiate([ snde.math_parameter_recording("/test_channel"), snde.math_parameter_double_const(scalefactor) ],
                                                               #[ snde.shared_string("scaled_channel2") ], # Note that this is relative to the context given below, which is "/"
                                                              # "/",
                                                              # False,
                                                              # False,
                                                              # False,
                                                              # snde.math_definition("c++ definition"),
                                                              # [],
                                                              # None)
#recdb.add_math_function(transact,scaled_channel_function2,False)

testchan_config=snde.channelconfig("/test_channel", "main",False)
  
testchan = recdb.reserve_channel(transact,testchan_config);

(transact.math["/scaled_channel2"],) = scalar_multiply_function("/test_channel",scalefactor)

# demonstrate alternative ways to create the recording

test_rec_32 = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)

globalrev = transact.end_transaction().globalrev_available()

transact2 = recdb.start_transaction(); # Transaction RAII holder
test_rec_64 = snde.create_ndarray_ref(transact2,testchan,snde.SNDE_RTN_FLOAT64)
globalrev2 = transact2.end_transaction().globalrev_available()


test_rec_32.rec.metadata=snde.immutable_metadata()
test_rec_32.rec.mark_metadata_done()
test_rec_32.allocate_storage([ rec_len ]);

test_rec_64.rec.metadata=snde.immutable_metadata()
test_rec_64.rec.mark_metadata_done()
test_rec_64.allocate_storage([ rec_len ]);

# locking is only required for certain recordings
# with special storage under certain conditions,
# however it is always good to explicitly request
# the locks, as the locking is a no-op if
# locking is not actually required.
# Note that requiring locking for read is extremely rare
# and won't apply to normal channels. Requiring locking
# for write is relatively common. 
locktokens = recdb.lockmgr.lock_recording_refs([
    (test_rec_32, True), # first element is recording_ref, 2nd parameter is false for read, true for write
    (test_rec_64, True),
],False)

for cnt in range(rec_len):
    test_rec_32.assign_double([cnt],100.0*math.sin(cnt))
    test_rec_64.assign_double([cnt],100.0*math.sin(cnt))
    pass

# must unlock prior to mark_data_ready
snde.unlock_rwlock_token_set(locktokens)

test_rec_32.rec.mark_data_ready()
test_rec_64.rec.mark_data_ready()

globalrev.wait_complete();
globalrev2.wait_complete();

scaled_rec_32 = globalrev.get_ndarray_ref("/scaled_channel")
scaled_rec_322 = globalrev.get_ndarray_ref("/scaled_channel2")

# verify it is OK to read these channels without locking
assert(not scaled_rec_32.ndinfo().requires_locking_read)
assert(not scaled_rec_322.ndinfo().requires_locking_read)
assert(not test_rec_32.ndinfo().requires_locking_read)

data_32 = scaled_rec_32.data
data_322 = scaled_rec_322.data

for cnt in range(rec_len):
    math_function_value = data_32[cnt]
    math_function_value2 = data_322[cnt]
    recalc_value = test_rec_32.data[cnt]*scalefactor
    print(" %f \t \t %f \t \t %f" % (recalc_value,math_function_value,math_function_value2)) 
    assert(abs(math_function_value-recalc_value) < 1e-4) # No functionality in Python to do single precision calculation for comparison
    assert(abs(math_function_value2-recalc_value) < 1e-4) # No functionality in Python to do single precision calculation for comparison
    pass

scaled_rec_64 = globalrev2.get_ndarray_ref("/scaled_channel")
assert(not scaled_rec_64.ndinfo().requires_locking_read)
assert(not test_rec_64.ndinfo().requires_locking_read)

data_64 = scaled_rec_64.data

for cnt in range(rec_len):
    math_function_value = data_64[cnt]
    recalc_value = test_rec_64.data[cnt]*scalefactor
    print(" %f \t \t %f" % (recalc_value,math_function_value)) 
    assert(math_function_value == recalc_value) 
    pass

