import time
import copy
import math
import multiprocessing
import spatialnde2 as snde

class measurement_time_monotonic(snde.measurement_time):
    value=None
    def __init__(self,value):
        super().__init__("")
        self.__disown__() # I think the shared_ptr's can handle the ownership in this context
        self.value = value
        pass
    def seconds_since_epoch(self):
        return value*1e-9
    def difference_seconds(self,to_subtract):
        return (self.value-to_subtract.Py_Downcast().value)*1e-9
    def __eq__(self,other):
        return self.value==other.Py_Downcast().value
    def __ne__(self,other):
        return self.value!=other.Py_Downcast().value
    def __lt__(self,other):
        #print("__lt__")
        #print("__lt__",self.value)
        #import pdb
        #pdb.set_trace()
        #print("__lt__",other.Py_Downcast().value)
        return self.value<other.Py_Downcast().value
    def __le__(self,other):
        return self.value<=other.Py_Downcast().value
    def __gt__(self,other):
        return self.value>other.Py_Downcast().value
    def __ge__(self,other):
        return self.value>=other.Py_Downcast().value
    #def __copy__(self):
        #return measurement_time_monotonic(self.value)
    def Py_Downcast(self):
        #return copy.deepcopy(self)
        return self
    pass

class measurement_clock_monotonic(snde.measurement_clock):
    def __init__(self):
        super().__init__("")
        pass

    def get_current_time(self):
        return measurement_time_monotonic(time.monotonic_ns())
    def sleep_for(self,time_seconds):
        time.sleep(time_seconds)
        pass
    pass

clock = measurement_clock_monotonic()

time1 = clock.get_current_time()
time.sleep(1)
time2 = clock.get_current_time()
print("should be true",time1==time1)
print("should be false",time1>time2)
print("time difference =",time2.difference_seconds(time1))

clock_cpp = snde.measurement_clock_cpp_system("")
time1_cpp = clock_cpp.get_current_time()
time.sleep(1)
time2_cpp = clock_cpp.get_current_time()
print("should be true",time1_cpp==time1_cpp)
print("should be false",time1_cpp>time2_cpp)
print("time difference =",time2_cpp.difference_seconds(time1_cpp))


#see if this actually works in the recording database
recdb=snde.recdatabase();
recdb.transmgr = snde.timed_transaction_manager(clock,0.25).upcast()
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
recdb.startup()

transact = recdb.start_transaction(clock.get_current_time());
testchan = recdb.define_channel(transact,"/test channel", "main");
test_ref = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)
globalrev = transact.end_transaction().globalrev_available()

test_rec_metadata = snde.constructible_metadata()
test_rec_metadata.AddMetaDatum(snde.metadatum_dbl("nde_axis0_inival",0.0));
rec_len=20
test_ref.rec.metadata = test_rec_metadata;
test_ref.rec.mark_metadata_done()
test_ref.allocate_storage([ rec_len ],False);

for cnt in range(rec_len):
    test_ref.assign_double([cnt],100.0*math.sin(cnt))
    pass

test_ref.rec.mark_data_ready()

globalrev.wait_complete();
