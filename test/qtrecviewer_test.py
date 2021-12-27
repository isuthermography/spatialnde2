import sys
import multiprocessing 
import math
import spatialnde2 as snde
from PySide2.QtWidgets import QApplication,QWidget,QMainWindow
from PySide2.QtCore import QCoreApplication,QObject,Qt

rec_len=100;

recdb=snde.recdatabase();
snde.setup_cpu(recdb,multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_math_functions(recdb,[])
recdb.startup()

 
transact = snde.active_transaction(recdb); # Transaction RAII holder

testchan_config=snde.channelconfig("/test channel", "main", recdb,False)
  
testchan = recdb.reserve_channel(testchan_config);

test_rec = snde.create_recording_ref(recdb,testchan,recdb,snde.SNDE_RTN_FLOAT32)

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


QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

app = QApplication(sys.argv)
window = QMainWindow()

viewer = snde.QTRecViewer(recdb,window)
window.setCentralWidget(viewer.QWidget())
window.show()
app.exec_()
