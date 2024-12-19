import sys
import pkgutil
import os.path
import multiprocessing 
import math
import numpy as np
import spatialnde2 as snde
# Check if SpatialNDE2 is built for Qt5 versus Qt6
spatialnde2_loader = pkgutil.get_loader("spatialnde2")
spatialnde2_qt_version=None
if spatialnde2_loader is not None:
    spatialnde2_path = spatialnde2_loader.get_filename()
    spatialnde2_dirpath = os.path.dirname(spatialnde2_path)
    compile_definitions_path = os.path.join(spatialnde2_dirpath, "compile_definitions.txt")
    with open(compile_definitions_path, "r") as fh:
        compile_definitions_str = fh.read()
        pass
    _spatialnde2_qt6_enabled = "-DSNDE_ENABLE_QT6=1" in compile_definitions_str
    if _spatialnde2_qt6_enabled:
        spatialnde2_qt_version="6"
        pass
    else:
        spatialnde2_qt_version="5"
        pass
    pass

if spatialnde2_qt_version == "5":
    try: 
        from PySide2.QtWidgets import QApplication,QWidget,QMainWindow
        from PySide2.QtCore import QCoreApplication,QObject,Qt
        from PySide2 import QtCore
        pass
    except ImportError:
        from PyQt5.QtWidgets import QApplication,QWidget,QMainWindow
        from PyQt5.QtCore import QCoreApplication,QObject,Qt
        from PyQt5 import QtCore
        pass
    pass
else:
    try: 
        from PySide6.QtWidgets import QApplication,QWidget,QMainWindow
        from PySide6.QtCore import QCoreApplication,QObject,Qt
        from PySide6 import QtCore
        pass
    except ImportError:
        from PyQt6.QtWidgets import QApplication,QWidget,QMainWindow
        from PyQt6.QtCore import QCoreApplication,QObject,Qt
        from PyQt6 import QtCore
        pass
    pass

rec_len=100;

recdb=snde.recdatabase();
snde.setup_cpu(recdb,[],multiprocessing.cpu_count())
snde.setup_storage_manager(recdb)
snde.setup_opencl(recdb,[],False,2,None)
snde.setup_math_functions(recdb,[])
recdb.startup()

 
transact = recdb.start_transaction(); # Transaction RAII holder

testchan_config=snde.channelconfig("/test channel", "main",False)
pointcloudchan_config=snde.channelconfig("/pointcloud channel", "main",False)
  
testchan = recdb.reserve_channel(transact,testchan_config);
pointcloudchan = recdb.reserve_channel(transact,pointcloudchan_config);

test_rec = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)
pointcloud_rec = snde.create_ndarray_ref(transact,pointcloudchan,snde.SNDE_RTN_SNDE_COORD3)

globalrev = transact.end_transaction().globalrev_available()

test_rec.rec.metadata=snde.constructible_metadata()
test_rec.rec.mark_metadata_done()
test_rec.allocate_storage([ rec_len ]);


nx = 12
ny = 10
x=np.linspace(-6,6,nx)
y=np.linspace(-5,5,ny)
(x_2d,y_2d)=np.meshgrid(x,y,indexing="ij")
r=9.0
# x^2 + y^2 + z^2 = r^2
# z^2 = r^2 - x^2 - y^2
z_2d = np.sqrt(r**2 - x_2d**2 - y_2d**2)

pc_metadata = snde.constructible_metadata()
pc_metadata.AddMetaDatum(snde.metadatum("snde_render_goal","SNDE_SRG_POINTCLOUD"))
pointcloud_rec.rec.metadata = pc_metadata
pointcloud_rec.rec.mark_metadata_done()
pointcloud_rec.allocate_storage([ nx,ny ],True);

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
    (pointcloud_rec,True)
],False)
for cnt in range(rec_len):
    test_rec.assign_double([cnt],100.0*math.sin(cnt))
    pass

pointcloud_rec.data["coord"][:,:,0]=x_2d
pointcloud_rec.data["coord"][:,:,1]=y_2d
pointcloud_rec.data["coord"][:,:,2]=z_2d
# must unlock prior to mark_data_ready
snde.unlock_rwlock_token_set(locktokens)

test_rec.rec.mark_data_ready()
pointcloud_rec.rec.mark_data_ready()

globalrev.wait_complete();


QCoreApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL)
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

app = QApplication(sys.argv)
window = QMainWindow()

viewer = snde.QTRecViewer(recdb,window)
window.setCentralWidget(viewer.QWidget())
window.show()
app.exec_()
