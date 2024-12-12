Using SpatialNDE2
=================

Initialization
--------------

Initialization of SpatialNDE2 involves creating the
recording database, setting up compute resources,
setting up storage resources, setting up math functions,
and calling the ``startup()`` method (C++)::
   
  #include "snde/recstore.hpp"
  #include "snde/recstore_setup.hpp"
  #include "snde/recstore_setup_opencl.hpp"
  
  ...
  
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  snde::setup_cpu(recdb,{"CPU"},std::thread::hardware_concurrency());
  snde::setup_opencl(recdb,false,4,nullptr); 
  snde::setup_storage_manager(recdb);
  snde::setup_math_functions(recdb,{});
  recdb->startup();

or in Python::

  import spatialnde2 as snde
  
  ...
  
  recdb=snde.recdatabase();
  snde.setup_cpu(recdb,["CPU"],multiprocessing.cpu_count())
  snde.setup_opencl(recdb,[],False,4,None)
  snde.setup_storage_manager(recdb)
  snde.setup_math_functions(recdb,[])
  recdb.startup()

Let's go through these lines one-by-one.
::
  
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();

This initializes the recording database object.
::
  
   snde::setup_cpu(recdb,{"CPU"},std::thread::hardware_concurrency());
   
This configures the CPU as a compute resource, allowing the
same number of threads as the number of CPU cores reported
by the operating system. The braces can be used to provide
a set of tags that can match the CPU to specific computations. 

::
   
  snde::setup_opencl(recdb,{},false,4,nullptr); 

This optional step enables OpenCL GPU acceleration. The second parameter
provides tags that can match to computations. The
third parameter is true to require that the primary GPU acceleration
device supports double precision floating point, false otherwise.
The 4th parameter specifies the maximum number of tasks to
simultaneously run on the GPU. The 5th parameter can be a pointer
to a string indicating a prefix to the platform name of the desired
OpenCL device, e.g. (C++)::
  
  std::make_shared<std::string>("Intel(R) OpenCL HD Graphics")

or (Python)::
  
  snde.shared_string("Intel(R) OpenCL HD Graphics")

If the platform prefix is specified as nullptr, the platform prefix
can alternatively be specified via the environment variable
``SNDE_OPENCL_PLATFORM``. Otherwise SpatialNDE2 will search for the
first OpenCL device that considers itself a GPU.

::
   
  snde.setup_storage_manager(recdb)

This step configures a default memory allocator, stored in ``recdb->lowlevel_alloc`` and a default storage manager, stored in ``recdb->default_storage_manager``. See ``recstore_setup.cpp`` for details.

::
   
  snde::setup_math_functions(recdb,{});

This provides a location to register additional math functions with the
recording database. They would be provided as the second parameter as
a sequence of (function name, ``math_function`` pointer pairs).

::
   
  recdb.startup()

This starts up the math threads and makes the recording database ready
to operate.

Defining Channels and Recordings
--------------------------------

Channels and recordings need to be defined within a transaction. For example (C++)::
  
  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction();  
  std::shared_ptr<snde::reserved_channel> testchan = recdb->define_channel(transact,"/test channel", "main");
  std::shared_ptr<snde::ndarray_recording_ref> test_ref = snde::create_ndarray_ref(transact,testchan,SNDE_RTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev_available();

or (Python)::

  transact = recdb.start_transaction();
  testchan = recdb.define_channel(transact,"/test channel", "main");
  test_ref = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)
  globalrev = transact.end_transaction().globalrev_available()

Each channel has a *path*, given as the second parameter to the
``define_channel()`` method of the ``recdatabase``.  The path uses
forward slashes ('/') as separator and roughly follows the general
POSIX filename conventions (although these paths are *not* filenames).
In this case the path of the newly created channel is "/test channel".

The third and fourth arguments of ``define_channel()`` are the name
and unique identifier of the channel owner. The unique identifier is
some arbitrary unique void pointer used to verify that recordings are
only created on a channel by the channel's owner. In most cases the
unique pointer will be the containing class instance. For channels
owned by the main function or an interactive session the convention is
to use the raw pointer to the recording database. This is accessed in
C++ from the shared pointer by the ``get()`` method or in Python from
the wrapped object by the ``raw()`` method.

The call to the ``create_ndarray_ref()`` function defines a new
``multi_ndarray_recording`` on ``/test channel`` containing a single
n-dimensional array with index 0 that is intended to hold 32 bit
floating point numbers (base type number identifiers are defined in
``recording.h``), and returns a recording reference to that single
n-dimensional array. It is also acceptible to pass
``SNDE_RTN_UNASSIGNED`` as the typenumber and then use the
``assign_recording_type()`` method to assign the type later.
The example code above does not yet place any data into the
recording. It just defines the existence. It is just fine to supply
the data later.

You need to be very careful to always eventually end the transaction
and mark the recording as ready. Otherwise, in certain common
situations such as the presence of self-dependent math channels or
globalrevision monitoring (``monitor_globalrevs`` object), all
subsequent recordings will be kept in memory waiting for this
recording to become ready so that it can be properly processed.
Depending on the rate of acquisition, leaving a recording unfinished
can potentially exhaust main memory very quickly.  Be sure to consider
the impact of an exception.

In C++ an exception will cause the transaction to end, as the
``active_transaction`` object will go out of scope. However, there is
no similar automatic marking of the generated recording(s) as
ready. Moreover in Python the loss of scope does not necessarily mean
immediate destruction (garbage collection might be delayed and/or the
variable might stay alive in an exeception stack backtrace). So an
explicit ``try...catch`` or ``try...except`` should be used to ensure
that a created recording is always marked as ready (even if empty).

Assigning Metadata
------------------

Metadata is applied to a new recording by creating a
``constructible_metadata`` object, adding metadata entries, and then
assigning to the ``metadata`` field of the recording at which point
the metadata is no longer mutable. Next the recording is marked as
having its metadata done (C++)::
   
  std::shared_ptr<snde::constructible_metadata> test_rec_metadata = std::make_shared<snde::constructible_metadata>();
  test_rec_metadata->AddMetaDatum(snde::metadatum("ande_array-axis0_offset",0.0));
  test_ref->rec->metadata = test_rec_metadata;
  test_ref->rec->mark_metadata_done();

or (Python)::
  
  test_rec_metadata = snde.constructible_metadata()
  test_rec_metadata.AddMetaDatum(snde.metadatum_dbl("ande_array-axis0_offset",0.0));
  
  test_ref.rec.metadata = test_rec_metadata;
  test_ref.rec.mark_metadata_done()

Each metadatum has a name, a type (integer,string, signed, unsigned) and
can be created with the functions

  * ``metadatum_int()``
  * ``metadatum_unsigned()``
  * ``metadatum_dbl()``
  * ``metadatum_str()``

Allocating the Array
---------------------------------------------
Space for the recordings array must be allocated
with the ``allocate_storage()`` method, (C++)::

  test_ref->allocate_storage(std::vector<snde_index>{rec_len},false);

or (Python) ::

  test_ref.allocate_storage([ rec_len ],False)

Pass multiple lengths to create a multi-dimensional array. The second
parameter, which defaults to false determines the storage layout for
multidimensional arrays. If false, the array will be stored with the
rightmost index selecting adjacent elements (row major, C style); if
true, the array will be stored with the leftmost index selecting adjacent
elements (column major, Fortran style).

Locking the Array
-----------------

For most storage and allocation managers locking is unnecessary, but
when writing code for general purpose applications or to be reused,
always lock arrays prior to access. For example::

  rwlock_token_set locktokens = recdb->lockmgr->lock_recording_refs({
    { test_ref, true },  
  }, false);

or (Python)::
  
  locktokens = recdb.lockmgr.lock_recording_refs([
    (test_ref, True),  
  ], False)

You provide a sequence of (recording reference, read/write) pairs
where the second element is false for read and true for write.  It is
important to lock all recordings in a single method call because that
way the locking code can ensure a consistent locking order is
followed. Multiple simultaneous read locks on a given array are
possible. Only one write lock can be held for a given array at a time,
and no read locks can exist in parallel with that write lock. The last
parameter to ``lock_recording_refs()`` indicates that you are locking it
for GPU access and is usually false.

The locks will last until explicitly unlocked or the containing
object is destroyed. See below for how to explicitly unlock.

Assigning Array Contents
------------------------

Floating point values can be assigned with the ``assign_double()``
method (C++)::
  
  for (size_t cnt=0;cnt < rec_len; cnt++) {
    test_ref->assign_double({cnt},100.0*sin(cnt));    
  }

or (Python)::
  
  for cnt in range(rec_len):
    test_ref.assign_double([cnt],100.0*math.sin(cnt))
    pass

In C++ if the reference is typed, you can use the ``element()``
method to obtain a writeable reference::

  test_ref_typed = std::dynamic_pointer_cast<snde::ndtyped_recording_ref<snde_float32>>(test_ref);
  for (size_t cnt=0;cnt < rec_len; cnt++) {
    test_ref_typed->element({cnt}) = 100.0*sin(cnt);    
  }

In Python, vectorized (numpy) access is also possible::

  test_ref.data[:] = np.sin(np.arange(rec_len),dtype='d') 

Unlocking the Array
-------------------

You must make sure the array is unlocked before marking the array as
ready (C++)::
  
  snde::unlock_rwlock_token_set(locktokens);

or (Python)::
  
  snde.unlock_rwlock_token_set(locktokens)

In C++ you can also ensure the array is unlocked by letting
the storing variable (``locktokens``, in this case) go
out of scope. In the future, the locktokens might be supported
as a Python context manager (``with`` statement) but that
is not implemented as of this writing.

Marking the Recording as Ready
------------------------------

The recording is marked as ready with the ``mark_data_and_metadata_ready()`` method of
the recording (C++)::
  
  test_ref->rec->mark_data_and_metadata_ready();

or (Python)::
  
  test_ref.rec.mark_data_and_metadata_ready()

Make sure all locks are released prior to calling the
``mark_data_and_metadata_ready()`` method.

Waiting for Globalrevision Completion
-------------------------------------

The ``end_transaction()`` method above returned a ``transaction``
object on which we called the ``globalrev_available()``
method to obtain a ``globalrevision``
object. That ``globalrevision`` may have math functions, data channels
from hardware devices, etc. that take time to become ready. Use the
``wait_complete()`` method to wait for all recordings in a particular
``globalrevision`` to be ready (or at least have metadata, for math
channels that only compute through metadata completion by default)
(C++)::
  
  globalrev->wait_complete();

or (Python)::
  
  globalrev.wait_complete()

Alternatively, instead of using ``globalrev_available()`` you can call ``globalrev()`` which automatically waits for completion. Just make sure that you have marked any recordings you have created as complete before waiting so that it doesn't wait forever.

Obtaining Globalrevisions
-------------------------

In the above example, a ``globalrevision`` was obtained from the
``end_transaction()`` method of an ``active_transaction``.  It is also
possible to obtain a globalrevision by calling the
``latest_defined_globalrev()`` or ``latest_globalrev()`` methods of
the ``recdatabase`` to obtain the most recently defined globalrevision
or the most recently fully complete globalrevision respectively.

If you want to see every new ``globalrevision`` that becomes
complete then you can call the ``start_monitoring_globalrevs()``
method of the ``recdatabase`` to obtain a ``monitor_globalrevs``
object. See the notification section of the concepts chapter for
more details.

You can always obtain the most recent complete global revision
with ``recdb.latest_globalrev()`` or the most recent defined
global revision (which may not yet be complete) with
``recdb.latest_defined_globalrev()``. Given a global revision
object stored in the variable ``globalrev``, you can list the
recordings in a global revision with ``globalrev.list_recordings()``
or the available n-dimensional array recording references with
``globalrev.list_ndarray_refs()``. Likewise you can obtain
a recording or an n-dimensional array reference with ``globalrev.get_recording()``
or ``globalrev.get_ndarray_ref()`` respectively. 

Pythonic Interface to SpatialNDE2
---------------------------------

The SpatialNDE2 Python bindings provide a simplified shorthand
"pythonic" interface to certain features. Specifically, quick ways
to access the latest global revision, access recordings and ndarray
references, and define math operations.

You can obtain the most recent complete global revision with
``recdb.latest`` (equivalent to ``recdb.latest_globalrev()``).
Given a global revision ``g``, you can obtain the list of recordings in
that global revision with ``g.rec``. Likewise, you can obtain the list of
ndarray references with ``g.ref``. You can then obtain the recording by
``g.rec[channel_path]``. Likewise, you can obtain the default ndarray reference in a recording with ``g.ref[channel_path]``, or a specific ndarray reference (indexed by integer array number or string array name) with ``g.ref[channel_path,index]``. Once you have a recording ``r``, you can list ndarray references within the recording by ``r.array``, and extract them with ``r.array[index]`` where index is the integer array number or string array name. Likewise, you can view metadata with ``r.metadata``.

As above, once you have an ndarray reference ``a``, you can obtain the recording with ``a.rec`` and the metadata with ``a.rec.metadata``. You can access the data with ``a.data``. For recordings under construction, you can modify the data with ``a.data[...]=`` or look at the layout information with ``a.layout``.

SpatialNDE2 ``active_transactions`` can be used as Python context managers via the ``with`` statement. For example, ::

  with recdb.start_transaction() as trans:
    new_ref = snde.create_ndarray_ref(transact,testchan,snde.SNDE_RTN_FLOAT32)
    new_ref.allocated_storage(my_data.shape,False)
    new_ref.data[...] = my_data
    new_ref.rec.metadata = snde.constructible_metadata()
    new_ref.rec.mark_data_and_metadata_ready()
    pass
  g = trans.globalrev()

In addition, there is a shorthand for defining math functions: ::

  with recdb.start_transaction as trans:
    trans.math["/avg"] = snde.averaging_downsampler("/raw",10,True)
    pass

For a single-result-channel math function outside of a transaction,
there is a similar shorthand that implicitly starts and ends
the transaction: ::
  
  recdb.math["/avg"] = snde.averaging_downsampler("/raw",10,True)

For either of the above shorthands, in addition to providing the
mandatory parameters to the math functions, you can also provide
keyword arguments: ``mutable`` (True/False enabling mutable math function output), ``execution_tags`` (a list of tag strings to match with compute resources), and ``extra_params`` (extra parameters passed to the math function).
   
.. _SNDEinDGPY:

Using SpatialNDE2 in Dataguzzler-Python
---------------------------------------

SpatialNDE2 provides Dataguzzler-Python include files for both non-GUI
creation of a recording database and creation of one or more QT
viewers. You can include one or both (the GUI include automatically
implies the non-GUI include with default parameters if you haven't
already explicitly included it).

To use, make sure you have the following defined in your ``.dgp``
configuration file for dataguzzler-python::
  
  from dataguzzler_python import dgpy
  import spatialnde2 as snde

  include(dgpy,"dgpy_startup.dpi") # If you get a NameError here, be sure you are executing this file with dataguzzler-python

  include(dgpy,"Qt.dpi",prefer_pyqt=False) 

  include(snde,"recdb.dpi",enable_opencl=True)
  include(snde,"recdb_gui.dpi")

The non-GUI configuration ``recdb.dpi`` has a single parameter
``enable_opencl`` that defaults to ``False`` but it is strongly
recommended you set to ``True`` if you have OpenCL compatible graphics
drivers or an add-on GPGPU.

The non-GUI configuration ``recdb.dpi`` defines four globals:

  * ``snde``, the spatialnde2 import, and
  * ``recdb``, the recording database object.
  * ``opencl_context``, the OpenCL context object, if available
  * ``opencl_devices``, A list of OpenCL devices, if available
    
The GUI configuration ``recdb_gui.dpi`` defines four more globals:
  * ``snde_RecViewerWindow`` A class derived from QT ``QMainWindow``
    that contains a ``QTRecViewer``.
  * ``viewer()``, a function for creating an additional viewer window
  * ``snde_recdb_windows``, a list containing all created viewer windows
  * ``main_viewer``, the viewer window that is automatically created by
    ``recdb_gui.dpi``. 

The QT windows operate in the main GUI thread and access from other
threads (including interactive consoles) is delegated to the main GUI
thread.
    

Creating an Interactive Viewer
------------------------------

The class ``QTRecViewer`` is a subclass of the QT ``QWidget`` that
implements the SpatialNDE2 interactive viewer. To create a viewer
in Dataguzzler-Python, use the ``recdb_gui.dpi`` include file as
described above.

The ``QTRecViewer`` can also be instantiated directly but generally
needs to be embedded in a QT ``QWindow`` (C++)::
  
  #include <QApplication>
  #include <QMainWindow>
  #include "snde/qtrecviewer.hpp"

  int main(int argc, char **argv)
  { 
    // (recording database initialization omitted) 
    
    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL); // Required by OpenSceneGraph library in some circumstances
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts); // Eliminate annoying QT warning message
    QApplication qapp(argc,argv);  
    QMainWindow window;
    
    QTRecViewer *Viewer = new QTRecViewer(recdb,&window);

    window.setCentralWidget(Viewer);
    window.show();
    
    qapp.exec();
  }
  
or (Python)::
  
  import spatialnde2 as snde
  from PySide2.QtWidgets import QApplication,QWidget,QMainWindow
  from PySide2.QtCore import QCoreApplication,QObject,Qt

  QCoreApplication.setAttribute(Qt.QtCore.Qt.AA_UseDesktopOpenGL) # Required by OpenSceneGraph library in some circumstances
  QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts) # Eliminate annoying QT warning message
  app = QApplication(sys.argv)
  window = QMainWindow()
  
  viewer = snde.QTRecViewer(recdb,window)
  window.setCentralWidget(viewer.QWidget())
  window.show()
  app.exec_()

Because QT requires that all GUI code run in the main thread and that
it occupy the main thread with its event loop, all processing after
the window is shown needs to either be performed from other threads or
by callbacks (such as QT timers or slots) from the QT event loop.  Do
not call methods of QT graphic objects (anything derived from QWidget,
including QTRecViewer) from threads other than the main thread.

Be aware that the QTRecViewer is a SWIG binding of a QT object. 

Python Bindings
---------------

The Python bindings are written using `SWIG <http://www.swig.org>`_.
Most C++ objects are wrapped with STL ``std::shared_ptr<>``
smart pointer templates and these are natively supported by SWIG so
there is little risk of trouble with object lifetimes. However it
is possible for the same C++ object to have multiple Python wrappers
so the Python "is" (object equivalence) operator should not be relied on
(instead you can compare the raw C++ pointers returned by the `raw()`
method).

Python function and method calls generally have the same names and
arguments as C++.  Some objects that use complicated C++ templates, or
which are generally not for external consumption may not be wrapped.

SWIG can natively assemble Python lists into ``std::vector<>``
templates and lists of pair tuples into ``std::vector<std::pair<>>``
and these are implemented in the bindings and should generally work
transparently for functions which need or return this type of data.
For example, the bindings define a ``snde.StringVector`` which is
really a ``std::shared_ptr<std::vector<std::string>>`` and a
``snde.StringPairVector`` which is really a
``std::shared_ptr<std::vector<std::pair<std::string,std::string>>>``

SpatialNDE2 has special code for translating certain objects into
forms compatible with particular Python packages. The relevant package
has to be installed and importable for this code to work. Specifically

  * The ``data()`` method of ``ndarray_recording_ref`` objects creates a NumPy
    array representation of n-dimensional array data.
  * Unlike most wrapped classes, the QTRecViewer class does **NOT**
    have its lifetime managed by an STL ``shared_pointer`` but rather
    by its QT parent object. The QTRecViewer class has ``QWidget()``
    and ``QObject()`` methods that return ``shiboken2``-wrapped
    pointers that are compatible with the PySide2 QT bindings. The
    SWIG-wrapper of the QTRecViewer does NOT own the underlying
    QTRecViewer object but rather requires a ``shiboken2``-wrapped
    ``QWidget`` as its parent that will take ownership and control
    the lifetime of the QTRecViewer. Any SWIG-wrapped references that
    still exist should no longer be used after the parent of the
    QTRecViewer destroys it.
  * OpenCL C++ objects get translated to/from `PyOpenCL <https://documen.tician.de/pyopencl/>`_ 

Care is required when interfacing between SWIG wrappers and native C++
code. Most SpatialNDE2 objects have a method
``produce_raw_shared_pointer()`` that returns the pointer to a newly
allocated ``std::shared_ptr<T>`` object in the form of a Python long
integer. The purpose of this method is to allow wrapped objects to be
converted back into C++ objects and used by external C++ code. Additional
persistent shared pointers can be created by initializing new ``std::shared_ptr<T>``
objects copying the one pointed to by the returned integer. The
class's ``from_raw_shared_ptr()`` static method creates, SWIG-wraps,
and returns such shared pointer objects. 

The underlying C++ object will be kept in memory at least until the
shared pointer object pointed to by the returned integer is destroyed,
either by the C++ ``delete`` operator or (preferably) by passing it to
the same class's ``consume_raw_shared_ptr()`` static method (which returns
a newly SWIG-wrapped shared pointer object that will safely expire
if ignored).

So in general for each call to ``produce_raw_shared_ptr()`` there should
be exactly one call to ``consume_raw_shared_ptr()`` but arbitrary copies
of the underlying shared pointer object can be made in between, and these will
live until all references of the underlying SpatialNDE2 class object expire. 


Troubleshooting
---------------

You can set environment variables to enable debugging/logging output.
Leave the environment variable unset, set to 0, or blank to disable;
debug output for a particular category. Anything else enables the
debugging output.  Debug categories:

  * ``SNDE_DC_RECDB`` The recording database 
  * ``SNDE_DC_RECMATH`` Math calculations of the recording database
  * ``SNDE_DC_NOTIFY``  Notifications of the recording database
  * ``SNDE_DC_LOCKING`` Locking of recordings
  * ``SNDE_DC_APP`` Application logging
  * ``SNDE_DC_COMPUTE_DISPATCH`` Dispatch of math functions to compute resources
  * ``SNDE_DC_RENDERING`` Graphics rendering
  * ``SNDE_DC_DISPLAY`` Display layout and positioning in the compositor
  * ``SNDE_DC_EVENT`` Event traversal from the GUI 
  * ``SNDE_DC_VIEWER`` The QT-based recording viewer
  * ``SNDE_DC_X3D`` Loading .x3d graphics files
  * ``SNDE_DC_OPENCL`` OpenCL-based GPU acceleration
  * ``SNDE_DC_OPENCL_COMPILATION`` Warnings from compiling OpenCL GPU compute kernels
  * ``SNDE_DC_MEMLEAK`` Debugging of memory leaks based on global revision reference loops
  * ``SNDE_DC_ALL`` Enables all of the above. 

In addition you can get additional logging from the OpenSceneGraph
(rendering) library by setting the environment variable
``OSG_NOTIFY_LEVEL`` with settings (from least notification to most
notification): ``ALWAYS``, ``FATAL``, ``WARN``, ``NOTICE``, ``INFO``,
``DEBUG_INFO``, ``DEBUG_FP``.

Debugging straight C++ applications is fairly straightforward, but
there a few tricks that can be very useful especially when SpatialNDE2 is
scripted from Python such as when using Dataguzzler-Python.

  * When debugging with GDB, you can run the debugger on the
    Python binary. Then give the path to ``dataguzzler-python``,
    the Dataguzzler-Python configuration file, and other parameters
    with GDB's ``run`` command. For example::
      
      $ gdb /usr/bin/python
      (gdb) run /usr/local/bin/dataguzzler-python my_config.dgp
  * GDB similarly supports attaching to a running process, for example::
      
      $ gdb /usr/bin/python
      (gdb) attach <dataguzzler-python process id>
      
  * For the above to work you may need to have the debug symbols for
    Python and other libraries installed. For Linux operating systems
    with of Red Hat lineage use ``dnf debuginfo-install
    <packagename>``. For Ubuntu and similar see `Debug Symbol Packages
    <https://wiki.ubuntu.com/Debug%20Symbol%20Packages>`_.
  * For MS Visual Studio you can attach to a running process (Debug
    menu).  Be sure to click the "Select" button to the right of the
    "Attach to" box before selecting the process. In the popup, make
    sure Native Code and Python Code are both checked. Once that step
    is completed you should be able to successfully attach to a running
    Python process, insert breakpoints, step through code, and
    troubleshoot crashes. (NOTE: It does not work to attach to the
    "dataguzzler-python" loader process itself from VS; instead attach to the
    Python process that is started by the loader).
  * To troubleshoot a deadlock, attach to the deadlocked process and
    search for two threads that are both waiting on a mutex. By examining
    the mutex structure it should be possible to identify the thread
    that owns the mutex. A deadlock is almost certainly a locking order
    violation. The `gdb-automatic-deadlock-detector <https://github.com/DamZiobro/gdb-automatic-deadlock-detector>`_ automates deadlock detection
    using GDB and Python.




    
  
