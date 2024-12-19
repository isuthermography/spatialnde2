import multiprocessing 
import math
import spatialnde2 as snde


rec_len=100;

recdb=snde.recdatabase();
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
recdb.startup()

 
transact = recdb.start_transaction();
testchan = recdb.define_channel(transact,"/test channel", "main");
test_ref = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)
globalrev = transact.end_transaction().globalrev_available()

test_rec_metadata = snde.constructible_metadata()
test_rec_metadata.AddMetaDatum(snde.metadatum_dbl("nde_axis0_inival",0.0));

test_ref.rec.metadata = test_rec_metadata;
test_ref.rec.mark_metadata_done()
test_ref.allocate_storage([ rec_len ],False);

# locking is only required for certain recordings
# with special storage under certain conditions,
# however it is always good to explicitly request
# the locks, as the locking is a no-op if
# locking is not actually required.
# Note that requiring locking for read is extremely rare
# and won't apply to normal channels. Requiring locking
# for write is relatively common. 

locktokens = recdb.lockmgr.lock_recording_refs([
    (test_ref, True), # first element is recording_ref, 2nd parameter is false for read, true for write 
],False)
for cnt in range(rec_len):
    test_ref.assign_double([cnt],100.0*math.sin(cnt))
    pass
# must unlock prior to mark_data_ready
snde.unlock_rwlock_token_set(locktokens)

test_ref.rec.mark_data_ready()

globalrev.wait_complete();

rec = globalrev.get_ndarray_ref("/test channel")

data = rec.data

# Demonstrate export to raw shared pointer and reconstruction
# from raw shared pointer:
rec2 = snde.ndarray_recording_ref.consume_raw_shared_ptr(rec.produce_raw_shared_ptr())


# verify it is OK to read these channels without locking
assert(not rec.ndinfo().requires_locking_read)
assert(not rec2.ndinfo().requires_locking_read)

assert((rec2.data == rec.data).all())

print(data)
