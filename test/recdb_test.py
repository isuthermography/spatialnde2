import multiprocessing 
import math
import spatialnde2 as snde


rec_len=100;

alignment_requirements = snde.allocator_alignment()
recdb=snde.recdatabase(alignment_requirements);

cpu_compute = snde.available_compute_resource_cpu(recdb,multiprocessing.cpu_count())
recdb.compute_resources.add_resource(cpu_compute) 
 
transact = snde.active_transaction(recdb); # Transaction RAII holder

testchan_config=snde.channelconfig("/test channel", "main", recdb,False)
  
testchan = recdb.reserve_channel(testchan_config);

test_rec = snde.multi_ndarray_recording.create_typed_recording(recdb,testchan,recdb,snde.SNDE_RTN_FLOAT32)

globalrev = transact.end_transaction()

test_rec.rec.metadata=snde.constructible_metadata()
test_rec.rec.mark_metadata_done()
test_rec.allocate_storage([ rec_len ]);

# locking is only required for certain recordings
# with special storage under certain conditions,
# however it is always good to explicitly request
# the locks, as the locking is a no-op if
# locking is not actually required.
# Note that requiring locking for read is extremely rare
# and won't apply to normal channels. Requiring locking
# for write is relatively common. 

locktokens = recdb.lockmgr.lock_recording_refs([
    (test_rec, True), # first element is recording_ref, 2nd parameter is false for read, true for write 
])
for cnt in range(rec_len):
    test_rec.assign_double([cnt],100.0*math.sin(cnt))
    pass
# must unlock prior to mark_as_ready
snde.unlock_rwlock_token_set(locktokens)

test_rec.rec.mark_as_ready()

globalrev.wait_complete();

rec = globalrev.get_recording_ref("/test channel")

data = rec.data()

# Demonstrate export to raw shared pointer and reconstruction
# from raw shared pointer:
rec2 = snde.ndarray_recording_ref.from_raw_shared_ptr(rec.to_raw_shared_ptr())


# verify it is OK to read these channels without locking
assert(not rec.ndinfo().requires_locking_read)
assert(not rec2.ndinfo().requires_locking_read)

assert((rec2.data() == rec.data()).all())

print(data)
