import multiprocessing 
import math
import spatialnde2 as snde


rec_len=100;
recdb=snde.recdatabase();

recdb.default_storage_manager = snde.recording_storage_manager_shmem();

cpu_compute = snde.available_compute_resource_cpu(recdb,recdb.compute_resources,snde.SNDE_CR_CPU,multiprocessing.cpu_count())
recdb.compute_resources.compute_resources.append(cpu_compute.this) #std::thread::hardware_concurrency()))
 
transact = snde.active_transaction(recdb); # Transaction RAII holder

testchan_config=snde.channelconfig("/test channel", "main", recdb,False)
  
testchan = recdb.reserve_channel(testchan_config);

test_rec = snde.ndarray_recording.create_typed_recording(recdb,testchan,recdb,snde.SNDE_RTN_FLOAT32)

globalrev = transact.end_transaction()

test_rec.metadata=snde.immutable_metadata()
test_rec.mark_metadata_done()
test_rec.allocate_storage([ rec_len ]);

for cnt in range(rec_len):
    test_rec.assign_double([cnt],100.0*math.sin(cnt))
    pass

test_rec.mark_as_ready()

globalrev.wait_complete();

rec = globalrev.get_recording("/test channel")

data = rec.cast_to_ndarray().data()

# Demonstrate export to raw shared pointer and reconstruction
# from raw shared pointer:
rec2 = snde.recording_base.from_raw_shared_ptr(rec.to_raw_shared_ptr())

assert((rec2.cast_to_ndarray().data() == rec.cast_to_ndarray().data()).all())

print(data)
