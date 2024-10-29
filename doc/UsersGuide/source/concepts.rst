..module:: sphinx.ext.mathbase
    :synopsis:

SpatialNDE2 Concepts
====================

Detailed API documentation is included in the separate programming
reference.  This chapter introduces the abstract concepts that underly
the various API classes and documents the theory of operation of
some of the processes. 

Recordings
----------

The recording is the central concept of SpatialNDE2. It represents
(usually versioned) data that is usually attached to a named
channel. Recordings can store arbitrary textual and numerical
metadata. The recording is implemented by the abstract
``recording_base`` class. The ``recording_base`` class has an ``info``
member which is the C ``struct snde_recording_base`` structure (or an
extension thereof) that includes the most critical attributes for
access from a future C API. There are various subclasses of
``recording_base``, some of which are discussed below.

Recordings are usually initialized empty. Then the recording gets any
metadata added and once the data is complete the recording is marked
as ready. Most recordings are defined as immutable to simplify the
access semantics (n.b. the mutability is for the underlying data
array; the recording data structure is generally treated as fixed once
the recording is marked as ready). The most common type of recording
is the ``multi_ndarray_recording`` discussed below. Often single
arrays within a ``multi_ndarray_recording`` are referenced using the
``ndarray_recording_ref`` class. 

New recordings are created with the ``create_recording<T>()``,
``create_anonymous_recording<T>()`` (used for recordings not part of a
``recording_set_state`` or ``globalrevision``), or
``create_recording_math<T>()`` (used for recordings created during
math function execution) template functions.

Simple n-dimensional array recordings can also be created with the
``create_typed_ndarray_ref<T>()``,
``create_anonymous_typed_ndarray_ref<T>()``, or
``create_anonymous_typed_ndarray_ref_math<T>()`` template functions; or
with the ``create_ndarray_ref()``,
``create_anonymous_ndarray_ref()``, or
``create_anonymous_ndarray_ref_math()`` non-template functions.


Recording Database
------------------

The recording database holds information about versioned collections
of recordings in *channels*. The recording database keeps track of how
the channel contents are updated (*globalrevisions*). All updates are
performed via *transactions*. The recording database is implemented in
the ``recdatabase`` class that is usually stored as a shared pointer
in the ``recdb`` variable. 

Channels
--------

A channel represents a potential sequence of recordings within the
recording database. Channels can be created and deleted within
transactions. The channel at any given instant has a particular owner,
which should be the only object that can give the channel a new
recording.

The value of a channel at any instant is represented by a
recording. The owner of the channel can update the recording during a
*transaction* of the recording database.  The channel is represented
by the class ``channel`` with the configuration at any instant
represented by the immutable class ``channelconfig``


Math functions
--------------

Math functions are transformations that operate on recordings from
channels in the recording database, generating dependent recordings in
their own channels. Math functions are defined by the
``math_function`` class, are instantiated with given parameters (class
``math_parameter``) via the ``instantiate()`` method of the
``math_function`` and added into the recording database during a
*transaction* by the ``add_math_function`` method of the recording
database. Math functions can be imported from a supplying module or package
using Python or looked up via the ``lookup_math_function`` method
of the recording database. 


Globalrevisions and Recording Set States
----------------------------------------

The ``recording_set_state`` is a class that represents a coherent set
of recordings corresponding to a conceptual instant in time. The
``globalrevision`` is a special case (subclass) of ``recording_set_state``
representing a particular numbered such global revision within a
recording database. Among other member variables the ``recording_set_state``
tracks the status of the contained recordings in ``recstatus`` and the
status of math functions in ``mathstatus``.

Just because a ``recording_set_state`` or ``globalrevision`` object exists
doesn't mean that all of the recordings within have their data or
that all of the math channels have finished executing. Call the
``wait_complete()`` method to wait for all recordings within to become
ready. 


Transactions
------------

The transaction represents a conceptually instantaneous grouping of
changes to a recording database. A transaction is started by
calling the ``start_transaction()`` method of the recording
database.

Within the transaction you can define new channels, add math
functions, create new recordings for channels, etc. The transaction is
ended by calling the ``end_transaction()`` method of the
``active_transaction`` object, which then returns a ``transaction``
object on which you can call the ``.globalrev()`` method that returns
a ``globalrevision``
object you can use to access the recordings within the
transaction. The ``end_transaction()`` also allows math functions
dependent on recordings that were changed within the transaction to
start executing once all of the recordings they are dependent on are
complete and ready. The ``.globalrev()`` method mentioned above waits for all recordings (including math calculations) to be complete. If you do not wish to wait for completion, you can instead call ``.globalrev_available()`` that only waits for existence of the global revision, not completion.

Multi-N-Dimensional-Array Recordings
------------------------------------
The ``multi_ndarray_recording`` is the most common recording class for
storing recorded NDE data. It represents a collection of n-dimensional
data arrays with common metadata. While most recordings will only
contain a single n-dimensional array, the need for multiple such
arrays is common enough (e.g. for graphics data structures, growing
recordings) to justify this as the base data structure.

The individual arrays within a ``multi_ndarray_recording`` are accessed
using the ``ndarray_recording_ref`` reference class (or a subclass thereof) returned by the ``reference_ndarray()`` method or by the ``get_ndarray_ref()`` method of the ``recording_set_state`` or ``globalrevision``. 
Note that the single-ndarray ``ndarray_recording_ref`` is **not** a subclass
of ``multi_ndarray_recording`` so that the different n-dimensional arrays
within a single ``multi_ndarray_recording`` can be accessed by their own
``ndarray_recording_ref`` objects. 

The ``multi_ndarray_recording`` has C++ STL vector members ``layouts`` and
``storage`` representing the memory layout and underlying memory storage
for each n-dimensional array. An optional set of ``name_mapping`` and ``name_reverse_mapping`` hash tables can be used to define names for the contained
n-dimensional arrays rather than just using indices. 

For access by C code, the ``info`` member of a
``multi_ndarray_recording`` points to an extended C structure ``struct
snde_multi_ndarray_recording`` that starts with the ``struct
snde_recording_base`` base structure.  The ``struct
snde_multi_ndarray_recording`` then points to multiple ``struct
snde_ndarray_info`` representing the indivdual arrays.

Fusion-N-Dimensional-Array Recordings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One subclass of the ``multi_ndarray_recording`` is the ``fusion_ndarray_recording``. This subclass is useful for input data that is recorded over time in the form of a weighted average.
The data in this subclass is represented by the equation: 
:math:`\sum\nolimits_{i=0}^{n-1}X_{i}w_{i}/\sum\nolimits_{j=0}^{n-1}w_{j}`.
The variable :math:`X_i` represents some measured input for scan iteration `i`, :math:`w_i`
represents the weight assigned to that input, and `n` represents the current total number of scans collected. 

The ``fusion_ndarray_recording`` is comprised of two sub-arrays called ``"accumulator"`` and ``"totals"``.
The ``"accumulator"`` sub-array, which represents the numerator of this equation, contains the sum of all measured values to be stored in the recording, multiplied by their associated weights, carried out to scan iteration `n-1`. 
The ``"totals"`` sub-array, which represents the denominator of this equation, stores the values of the sum to scan iteration `n-1` of the weights assigned to each previous scan iteration.

When rendering this datatype, the SpatialNDE2 viewer can render 2D images representing the quotient of the accumulator and totals, which is the weighted average. Rendering of the 
``fusion_ndarray_recording`` subclass is specially handeled by the ``fusion_ndarray_recording_display_handler`` class defined in ``display_requirements.hpp``. Colormaps for real ``fusion_ndarray_recordings`` are 
generated as normal according to the selected colormap in the viewer, except the blue channel is overridden and represents the ratio of the total weighting for a particular pixel to the maximum total weighting for 
any pixel in the 2D image. Colormaps for complex ``fusion_ndarray_recordings`` are generated with red and green channel values determined, respectively, by the real and imaginary components of the weighted average. 
The blue channel is determined the same way as for the real case.
 
.. _GeometricObjects:

Geometric Objects such as Parts and CAD Models
----------------------------------------------

Geometric objects can be loaded using functions such as
``x3d_load_geometry()`` which takes the filename, index of the
shape within the file, and other parameters including ``processing_tags``.
Each ``processing_tag`` is a string representing some sort of pre-processing
that should be done either as part of the loading process, or by defining
a math function to store an additional output. The ``x3d_load_geometry()``
function accepts two preprocessing tags: ``reindex_vertices`` and ``reindex_tex_vertices`` which can enable reindexing during the loading process. In addition
math functions can register additional postprocessing tags, such as ``trinormals``, ``inplanemat``, ``projinfo``, etc. which will then trigger automatic
instantiation of the relevant math function as the last step in the loading
process. Currently implemented processing tags include: 


  * ``reindex_vertices`` (x3d loader only): Reindex the mesh vertices
    to create a new connectivity graph rather than relying on
    connectivity information in the loaded file.
  * ``reindex_tex_vertices`` (x3d loader only): Reindex the
    parameterization (texture) mesh vertices to create a new
    connectivity graph rather than relying on connectivity information
    in the loaded file.
  * ``trinormals``: Generate per-triangle normal vectors. 
  * ``inplanemat``: Generate per-triangle in-plane coordinate systems (requires ``trinormals``)
  * ``projinfo``: Generate per-triangle transforms between in-plane coordinates and uv parameterization (requires parameterization (texture) coordinates and ``inplanemat``).
  * ``boxes3d``: Generate triangle mesh bounding box octtree used for raytracing (requires ``trinormals`` and ``inplanemat``). 
  * ``boxes2d``: Generate uv triangle mesh bounding box quadtree used for mapping from uv coordinates to 3D location (requires parameterization (texture) coordinates). 
    
Loaded geometric objects end up represented as a collection of arrays,
typically a sub-tree of recordings, most or all of which are stored
using a ``graphics_storage_manager``. The sub-tree itself (if loaded
from disk) has a ``loaded_part_geometry_recording`` as its root.
Within, there is a ``meshed`` recording of class ``meshed_part_recording``,
which is a ``multi_ndarray_recording`` subclass that contains a single
1D array with a single element of ``struct snde_part``. There may also
be a ``uv`` of class ``meshed_parameterization_recording`` representing the
surface parameterization (texture coordinates), a ``texed`` recording of
class ``textured_part_recording`` and possibly one or more recordings
containing texture data.

The image below shows the included plate and probe 3D models in the
SpatialNDE2 viewer, listing some of the various recordings on the left that
are used for rendering and ray tracing.

.. image:: plate_and_probe.png
   :width: 800
   :alt: Screenshot of viewer window with plate and probe 3D models.


The ``graphics_storage_manager`` stores geometric objects in
a set of shared arrays pointed to by the ``struct snde_geometrydata``.
Space in the arrays is an ``allocation`` reserved by an ``allocator``.
Some arrays are allocated directly; others are *followers* which follow
the allocation pattern of another array.

For example the ``parts`` array of ``struct snde_part`` represents the
various discrete boundary-represented (BREP) models of physical parts.
Each part has both topological representation (represented by
the ``first_topo`` and ``num_topo`` field which index into the ``topos``
array of ``struct snde_part``) and geometrical
representation (the various triangles and vertices fields). 

A more detailed discussion of graphics and geometric objects
is planned for another chapter. 

.. _OrientationsAndPoses: 

Orientations and Object Poses
-----------------------------
"Pose" is a technical term referring to the rotation and position
of an object in three-dimensional space. Within the context
of SpatialNDE2, we will measure and store the "pose" of an object
as the transform (an snde_orientation3, with Numpy dtype
representation ``[('quat', '<f4', (4,)),('offset', '<f4', (4,))]``)
that, when multiplied on the right by a position or vector in
object coordinates, gives the position or vector in world coordinates.

Within the context of SpatialNDE2, an *orientation* is a relation
(rotation **and** translation) between two coordinate frames,
represented as a ratio. 
The orientation of coordinate frame A relative to coordinate frame B,
perhaps referred to as ``orient_A_over_B``, when multiplied on the
right by a position or vector in B coordinates gives the position
or vector in A coordinates. Thus the "Pose of A" is equivalent to ``orient_world_over_A``.

We can then use dimensional analysis to construct a desired orientation
or pose from pieces. However since left and right multiplication are
different, the order matters. In general if you have an ``_over_A``
it should be multiplied on the right by either coordinates relative to A
or an ``orient_A_over_``. 

The underlying implementation, while represented by an offset and quaternion,
is designed to behave equivalently to 4x4 transformation matrices in
Homogeneous (projective) coordinates as commonly used in computer
graphics, with the ``quat`` equivalent to the upper 3x3, and the
offset being the rightmost column (except we define the last entry in the
offset to be always zero, whereas in the matrix representation it would
be always one). Thus when you multiply an orientation by a position,
it first applies the rotation ``quat`` and then adds the offset. Multiplying
an orientation by a vector applies the rotation and ignores the offset.
These multiplication operations are implemented in ``quaternion.h`` by
``orientation_apply_position()`` and ``orientation_apply_vector()``,
respectively. 

N-Dimensional-Array Recording References and Typed Recording References
-----------------------------------------------------------------------

The ``ndarray_recording_ref`` is the previously mentioned class for
referencing a single n-dimensional array within a
``multi_ndarray_recording``.  It can be obtained via the
``reference_ndarray()`` method of the ``multi_ndarray_recording`` or
the ``get_ndarray_ref()`` method of a ``recording_set_state`` or
``globalrevision``. The ``ndarray_recording_ref`` contains a C++ STL
shared pointer to the recording, ensuring that the recording will be
kept in memory as long as the reference exists. It also contains an
index indicating which n-dimensional array within the recording is
being referenced. In addition, the ``nd_array_recording_ref`` contains
deep references to the recording state, ndarray layout entry, and
ndarray storage entries of the ``multi_ndarray_recording``. As these
references may be invalidated if the ``std::vector`` s in
``multi_ndarray_recording`` grow, it is important to wait until all
n-dimensional arrays have been created within a given recording prior
to creating references.

You can call the ``void_shifted_arrayptr()`` method to get a pointer
to the data array itself (the shift part relates to use in graphics
arrays where multiple recordings share the same storage pool) and you
can call the ``element_dataptr()`` method to get a pointer to a
particular element.  For compatible arrays you can use the
``element_double()`` method to read an element as double precision
floating point or ``assign_double()`` to modify an element as double
precision floating point. There are similar methods for 64-bit signed
and unsigned integers: ``element_int()``, ``assign_int()``,
``element_unsigned()``, and ``assign_unsigned()``.

Be warned if you use ``void_shifted_arrayptr()`` that the layout of
data within the n-dimensional array is not necessarily contiguous. You
can use the ``layout->is_contiguous()``,
``layout->is_f_contiguous()``, and/or ``layout->is_c_contiguous()`` to
check for wether the array layout is or is not contiguous, follows
Fortran indexing conventions, and/or follows C indexing conventions
respectively. If your code makes contiguity assumptions they need to
be tested!


There is also a type-specific subclass ``ndtyped_recording_ref<T>``
you can cast the ``ndarray_recording_ref`` to using
``std::dynamic_pointer_cast<T>()``. The cast returns nullptr in the
case of a type mismatch, but otherwise you get a reference subclass
that is specific to the type of the data contained in the particular
n-dimensional array. You can then get a pointer to the first element
with the ``shifted_arrayptr()`` method (see warning above about array
layouts) and in this case *reference* elements with the ``element()``
method.

Notification
------------

While math notification is handled internally to the recording
database, at times you may need to be able to get notified when
new data on a channel becomes ready, when a new globalrevision
is available, etc.

Within the context of a specific ``recording_set_state``, the class
``channel_notify`` will send notifications based on
``channel_notification_criteria`` which include particular channel(s)
having their metadata ready or becoming fully ready, or the entire
``recording_set_state`` becoming complete. The ``channel_notify`` is
applied to a specific ``recording_set_state`` and therefore all
recordings within that set should still be valid so long as the
``recording_set_state`` remains valid, but there is no guarantee that
mutable channels will not change.

If you want an update whenever a new ``globalrevision`` becomes
complete then you can call the ``start_monitoring_globalrevs()``
method of the ``recdatabase`` to obtain a ``monitor_globalrevs``
object. You can specify the first globalrev you are interested in
(which defaults to the result of the ``latest_globalrev()`` method)
and a boolean flag, which if set will inhibit writes to mutable
recordings within the globalrevision until your monitoring object is
done with it.

You can then loop over the ``wait_next()`` or
``wait_next_inhibit_mutable`` methods of the ``monitor_globalrevs``
object. This gives you the ability to look at every new globalrevision
in turn.

Note that it is critically important to call the ``close()`` method of
the ``monitor_globalrevs`` object if you stop looping for any reason,
including some kind of exception or error. Usually any code after
``start_monitoring_globalrevs()`` should be wrapped with an exception
handler.  Otherwise memory usage could accumulate very rapidly as
recordings are held in memory on behalf of your monitoring loop and
never freed.


Storage
-------

Some sort of physical storage location is required for each
n-dimensional array. Storage is managed by a
``recording_storage_manager`` which in turn uses a ``memallocator`` to
provide the low level physical storage layer. The
``recording_storage_manager`` returns a ``recording_storage`` when an
allocation is requested. 

The purpose of the storage manager abstraction is to allow multiple
recordings to be stored in shared arrays so that the shared array can
be passed to a GPU as a single parameter, thus allowing GPU codes to
operate on dynamic collections of recordings rather than single
recordings.  One use case would be for the geometries of a collection
of objects under simultaneous inspection. The ``geometry_storage_manager``
implements such an approach for storing object geometry data. By
comparison the ``recording_storage_manager_simple`` delegates directly
to the underlying ``memallocator``.

Storage managers are defined for channels, and apply to that channel and any sub-channels. Note that if a parent channel changes its storage manager, there may be some latency before the sub-channels start using it. In particular, any transactions started before the transaction and the parent channel is realized. For this reason, when switching storage managers, always end the transaction and wait for realization with the ``.globalrev_available()`` or ``.globalrev()`` methods.

Memory Allocators
-----------------

The ``memallocator`` abstraction provides low level allocation service
to the storage manager. It also provides an API to obtain a nonmoving
copy or reference to the allocated data. In cases such as the
``geometry_storage_manager`` where arrays are shared across multiple
recordings, the underlying storage array may need to be resized (grow)
as new recordings are created. The ``nonmoving_copy_or_reference`` is
a copy or reference to a segment of a memory space that is guaranteed
to stay at a fixed address. It is defined by class ``recording_storage_reference``.

A simple ``cmemallocator`` uses the standard ``malloc()``, ``free()``,
etc. calls to provide that functionality. It does not support
nonmoving references, so if a nonmoving copy or reference is
requested, it generates a copy.

Other memory allocators can provide additional and more sophisticated
functionality. For example, the ``shared_memory_allocator_posix`` uses
the POSIX shared memory API to store recordings. These recordings can
then be accessed by other processes, creating a pathway for high performance
inter-process communication. In addition ``shared_memory_allocator_posix``
can use the operating system's virtual memory subsystem to obtain
a nonmoving reference to an allocation that might move around due to
reallocation, thus saving the space and performance degradation
involved in creating a copy. 


Compute Resources
-----------------

On setup the recording database is configured with multiple instances
of ``available_compute_resource`` in its
``available_compute_resource_database``. Each
``available_compute_resource`` represents perhaps a set of CPU threads
or a GPU device or similar.

An ``instantiated_math_function``, generates a list of
``compute_resource_option`` instances in the
``perform_compute_options`` phase of its execution. The
``compute_resource_option`` instances indicate different possible ways
to execute the ``instantiated_math_function`` and rough estimates of
the resources required for each. For example the math function can
provide both a ``compute_resource_option_cpu`` that offers to execute
strictly on CPU along with a ``compute_resource_option_opencl`` that
uses primarily GPU based compute.

The math engine then selects particular CPU threads (with a maximum
number of cores to use) and GPU devices (if applicable) to execute the
math function and provides an ``assigned_compute_resource`` to pass
that information back to the math function.

Math Function Objects
---------------------

A math function that is available for use is defined by a C++ STL
shared pointer to a class ``math_function``. For math functions
implemented in C++ the ``math_function`` object is generally created
during static initialization of a particular DLL/shared object and
immediately stored in a registry via the ``register_math_function()``
function. The math function is generally named according to a
Python-style package and module path
(e.g. ``spatialnde2.averaging_downsampler``). Ideally a SWIG wrapped
copy of the math function should also be available via a Python
import of the same path.

The recording database also maintains a map of addon math functions
that superimposes over the static initialization registry when
accessed by the ``lookup_math_function()`` method of ``recdatabase``.

The ``math_function`` object contains basic information about the
parameters of the math function and the nature of the resulting
output, along with a virtual method ``instantiate()`` which
instantiates the function with particular parameters and result
channels, and an ``initiate_execution`` lambda that creates the
``executing_math_function`` object which will track the execution of
this function within a particular ``recording_set_state`` once it has
been determined that the function may need to execute within that
``recording_set_state``.

Math Function Instantiation
---------------------------

Math function instantiation is the process of defining a particular
set of parameters (subclasses of ``math_parameter``, including
channels specified by channel path, constants, and more) and output
channel paths. The ``instantiated_math_function`` once created is
immutable, but still needs to be assigned into the recording database
during a transaction using the ``add_math_function`` method of the
recording database


Math Function Execution
-----------------------

Math functions execute in the context of a consistent set of
recordings, the ``recording_set_state`` (which in most cases is the
``globalrevision`` subclass). The ``math_function_execution`` is
created once it is clearly plausible that the math function might need
to execute. Generally, if any recordings the function is dependent on
have changed, or if indirect dependences added by the
``find_additional_deps()`` lambda of the ``math_function`` have
changed, then a ``math_function_execution`` will be created. The
``math_function_execution`` can be referenced by subsequent
``recording_set_state`` or ``globalrevision`` instances if none of the
recordings the function is dependent on change.

Once all of the recordings the function is dependent on in this
``recording_set_state`` have become ready, an
``executing_math_function`` is created via the
``initiate_execution()`` lambda of the ``math_function``.  This
``executing_math_function`` is referenced by the
``math_function_execution`` and tracks the steps involved in executing
the math function. The steps are:

  * If the function is ``new_revision_optional``, deciding whether or
    not to execute (``decide_execution()``)
  * Providing a list of compute options (``perform_compute_options()``)
  * Defining the output recordings (``define_recs()``). At this point
    the ``executing_math_function`` will have a valid ``compute_resource``
    and ``selected_compute_option`` members assigned.
  * Assigning metadata (``metadata()``)
  * Performing locking/allocation (``lock_alloc()``)
  * Performing the execution (``exec()``).

The steps are executed in order. ``decide_execution()`` and
``perform_compute_options()`` are executed in an arbitrary thread by
the main CPU. After ``perform_compute_options()`` the
``math_function_execution`` is queued as a ``pending_computation`` and
once a suitable ``available_compute_resource`` is available, it is
assigned into the ``executing_math_function``, and the thread corresponding
to the CPU portion of that ``available_compute_resource`` is dispatched
to execute ``define_recs()`` and the subsequent methods. If the math function is
``metadataonly`` and nothing has requested actual data, the execution
stops after assigning metadata (but may restart later if actual data
is requested). 

By the end of the execution function, it should have marked metadata
as done on all output recordings (``mark_metadata_done()`` method of
the recording) and the data as being ready (``mark_data_and_metadata_ready()`` method).

If math code throws an exception, it will be caught and (if the
exception was an ``snde_error()``) a backtrace printed. Exceptions of 
other types may not print a backtrace; it may be helpful in that
case to disable exception handling by rebuilding spatialnde2 with the
``SNDE_RCR_DISABLE_EXCEPTION_HANDLING`` preprocessor symbol defined.
With ``SNDE_RCR_DISABLE_EXCEPTION_HANDLING`` the exception will instead cause
an immediate crash, which may generate a core dump or drop into the debugger, depending on your system configuration. This can make it easier to debug the exception.
environment and

Math Function Messages
----------------------

Some math functions have the ability to receive messages.  Such messages can
be used to reset the state of a math function, trigger a special process, provide
new information outside of the normal recording process, etc.  The message is sent
in a transaction using the ``send_math_message()`` function.  The 
``instantiated_math_function`` object returned by the ``instantiate()`` method
is required, as well as a string key name defining the message and the message
value.  The message value is any ``math_instance_parameter``. 

Threading and Locking
---------------------

SpatialNDE2 objects are generally thread safe and SpatialNDE2 is
designed to be used in an aggressively threaded environment. The primary
approach to avoiding race conditions between threads is defining objects
and data structures as semantically immutable either once constructed or
once a particular step has been performed. The secondary approach is the use
of locking and/or atomic variables.

To prevent deadlocks, all locks must be acquired following a
particular (partial) order. The overall locking order is defined in the
comments at the start of ``lockmanager.hpp``. Be aware that any unique
resource that a thread can acquire and that another thread would need
to wait for can act like a lock and may need to be considered as part
of the locking order. For example, only one thread can start a
transaction in the recording database at a time. Thus entry into a
recording database transaction acts like a lock and must be included
in the locking order. 

Many classes have an ``admin`` ``std::mutex`` that must be held to
read or write portions or all of the class members. The class
definition will usually have comments indicating where the ``admin``
lock falls within the locking order of the SpatialNDE2 library.
The class definition comments will also indicate whether certain
members are considered immutable once constructed, meaning generally
safe to read from all threads once suitable notification has
propagated.

It is also important that locks from other tools or libraries in use
also be considered as part of the locking order. For example, the
Python global interpreter lock (GIL). The SWIG-generated Python
bindings of SpatialNDE2 automatically drop the GIL on entry into
SpatialNDE2 code. However, any call that might somehow directly
or indirectly call Python code could reacquire the GIL. From the
perspective of SpatialNDE2, the GIL is treated as a "last" lock,
i.e. you are not allowed to acquire any other lock while holding
the GIL, but the flip side is you are free to acquire the GIL at
any time.

How does this work with locks early in the locking order such as
starting a transaction, which may want to be done from Python code?
Simple: Because the transaction initiation is SWIG-wrapped C++, the
SWIG wrapper will have dropped the GIL before the C++ attempts to
initiate the transaction. When the SWIG-wrapper returns, it reacquires
the GIL, which is fine because the GIL is at the end of the locking
order. As long as all calls into the SpatialNDE2 C++ library drop
the GIL, all is fine and there is no locking order violation.

There is a prospective risk if external C++ libraries that do not always
drop the GIL call directly or indirectly into SpatialNDE2. The
biggest risk is probably QT. `Significant work has been done
in PySide recently <https://www.qt.io/blog/qt-for-python-5.15.0-is-out>`_
on threading but since QT does not always drop the GIL there is a risk. 

It should also be noted that per QT guidelines QT widgets are only
accessible from the "main thread" of the application. That means the
QT recording viewer (class ``qtrecviewer``) must be created in the
main thread and that method calls to it must generally be performed
only from the main thread. When used from dataguzzler-python via
``recdb_gui.dpi``, the ``dataguzzler_python.QtWrapper`` class helps
ensure that all accesses are from the main thread by proxying method
calls to a dispatch loop running in the main thread. 



Locking of Recording Data Arrays
--------------------------------

Certain data arrays may need to be locked prior to reading or writing.
For example, consider a mutable array. While the ordering inherent in
the math logic protects math function access from interference by
other math functions, external access needs to be protected.

In addition certain memory allocators or storage managers might
require locking prior to read and/or write access to a data array. One
example would be OpenCL GPU read access to a portion of a graphics
storage array.  Per the OpenCL specification simultaneous write access
to the full array buffer and read access to the portion's sub-buffer
triggers undefined behavior even if the read and write do not
overlap. Thus the write to the full buffer must be prevented while the
sub-buffer is being read, and this means that both read and write
accesses require appropriate locking. 

To give a another example, consider a future storage manager that keeps
the only copy of an array on-GPU. In order to read this data from the
CPU, some mapping operation that maps the GPU memory into CPU
address space would be needed prior to CPU read or write access, and
this mapping could potentially be triggered by the locking attempt. 

Other cases where locking may be required involve mutable arrays.  A
dependent math function that accesses the an array given as a math
function parameter can usually safely read its parameter because the
math logic will enforce sequential execution.  Likewise or the math
function that updates the array can usually safely read and write it.
However, other code in other threads cannot usually safely read a
mutable array without locking.

For all of these reasons it is strongly recommended that all array
reading and writing code lock the array(s) prior to access. See
the locking order documentation in ``lockmanager.hpp`` for detailed
ordering information, but the data array locks generally follow
the transaction, recording database, globalrevision, and recording
set state admin locks. Array locking will do nothing for arrays
that do not in fact require locking, and so long as multiple arrays
are locked in a single call, automatic correct ordering of those locks
is guaranteed.

Use the ``lock_recording_refs()`` or ``lock_recording_arrays()``
convenience functions to perfrom such locking in most cases.  A slightly
more involved procedure is required for allocating ``graphics_storage``
components; see ``x3d.hpp`` for an example of that process. 


Caching of Data Arrays on GPUs
------------------------------

The ``openclcachemanager`` keeps track of what recording data has been
transferred to the GPU to minimize unnecessary copies. It also keeps
track of what array regions have been modified by the GPU and need
to be transferred back into main memory. 

The ``openclcachemanager`` keeps a map of weak STL shared pointers
to the ``openclcacheentry`` subclass of class ``cached_recording``.
The primary (strong) pointers are kept in the ``cache`` map within
the recording's ``recording_storage`` so that when references
to the ``recording_storage`` expire, the OpenCL ``cl::Buffer``
object within will be automatically released, freeing the on-GPU
buffer. In addition the destructor of the ``recording_storage`` 
calls the ``notify_storage_expiration`` of the ``cachemanager`` so that
the ``cachemanager`` can clean up its tables. 

The ``openclcacheentry`` keeps track of ``invalidity``, where the GPU
copy is out of date compared to the CPU copy (triggered by
``recording_storage::mark_as_modified()``, and ``_dirtyregions``,
where the CPU copy is out of date compared to the GPU copy (triggered
by ``OpenCLBuffers::BufferDirty()``.

You access the GPU cache by creating an ``OpenCLBuffers`` object,
being sure to pass the suitable lock tokens returned by
``lockmanager::lock_recordings()`` or
``lockmanager::lock_recording_refs()``. The ``AddBufferAsKernelArg()``
method defines an OpenCL buffer or sub-buffer representing a
particular n-dimensional array selected from a
``multi_ndarray_recording``, from an ``ndarray_recording_ref``, or
from a ``recording_storage``, and sets that buffer as a numbered
argument to an OpenCL kernel.  The ``AddBufferAsKernelArg()`` method
also makes sure that any necessary transfers of data from main memory
to the GPU are properly queued and keeps track of the completion
events within the ``OpenCLBuffers`` object.

When it comes time to call the kernel, you can then pass the STL
vector of completion events returned by
``OpenCLBuffers::FillEvents()`` as the ``event_wait_list`` parameter
to ensure that the kernel will not execute until all necessary data
has been transferred to the GPU.

After calling the kernel you call ``OpenCLBuffers::BufferDirty()`` to indicate that
a kernel has made GPU-side modifications to a buffer. Finally 
call ``OpenCLBuffers::RemBuffers()`` with events (both usually the
kernel completion event) indicating when the input data is no longer needed
and when the output data is complete. This triggers the transfers
of dirty regions back to the CPU. Generally you want to wait
for completion so that you don't release your write lock (if applicable)
or mark the recording as ready until the transfer is complete.


When the ``RemBuffers()`` method copyback is complete it calls the
``mark_as_modified()`` method of ``recording_storage`` with itself
as the ``already_knows`` cachemanager so any other GPU devices caching
the same recording also get updated.

Recording Viewer
----------------

Viewer functionality is encapsulated in the ``QTRecViewer`` class,
which is a QT ``QWidget`` that is usually given its own window. The
viewer manages display of a user-selectable set of recordings,
selected via the ``QTRecSelector`` on the left hand side. The
``qtrec_position_manager`` manages the sliders, widgets, and events
controlling positions within the main view pane, updating the class
``display_info`` which tracks the the scaling, zoom, etc.  of the
various channels within the recording database. Because all QT
widgets should generally exist and be accessed solely from the
process main thread (GUI thread), the ``QTRecViewer`` likewise should
only be created and accessed from the process main thread. 

The ``display_info`` tracks channel-specific information in class
``display_channel`` with axis positions tracked via class
``display_axis`` and units via class ``display_unit``. The main view
pane is rendered by the ``qt_osg_compositor`` which is a QT
specialization of the more general ``osg_compositor`` that generates
the final render by compositing pre-renders of the various enabled
channels. The ``qt_osg_compositor`` has an option to enable threading,
which puts the wait for any on-demand prerequisite calculations in a
different thread, reducing contention for the main thread. There is a
second option to enable threaded OpenGL, which also puts the pre-renders
in a different thread. The final compositing is always performed in
the main thread.

Rendering Process
-----------------

The rendering process starts with performing any on-demand recording
math calculations (such as any render-specific data transforms, or
in at least some circumstances colormapping. The function
``traverse_display_requirements()`` looks at a set of ``display_channel``
objects that are to be rendered and performs a recursive traversal
to determine any ondemand calculations that need to be performed, the
geometric bounds for the rendering area, and the identity of the
renderer. The traversal must be recursive to, for example, identify
texture, geometry, and parameterization of a ``textured_part_recording``.
Every step of the traversal has a goal, which defaults to ``SNDE_SRG_RENDERING``.
A recording can set an alternative initial goal with the string metadata entry
``snde_render_goal``, which can be used to activate an alternative
visualization. If the alternative goal is relevant to 3D rendering, the
string metadata entry ``snde_render_goal_3d``, which can activates an
alternative visualization in cases where a 3d rendering process is
explicitly required by the containing step. 

The traversal is performed by looking up a
``registered_recording_display_handler`` based on the given goal and
the recording's particular subclass of ``recording_base``. The handler
is then instantiated and its ``get_display_requirement()`` method is
called, which can recursively traverse into other recordings that may
be required with the same or other goals.  The net result is a
recursive structure of ``display_requirement`` objects for each
channel provided to ``traverse_display_requirements()``.

The recursive ``display_requirement`` structure is then used to update
a ``recstore_display_transforms`` object which keeps track of the on-demand
math calculations required for rendering. The ``update()`` method of the
``recstore_display_transforms`` flattens the recursive ``display_requirement``
structure, merging identical sub-requirements, and creates an on-demand
math operation to execute any needed transforms into a new
``recording_set_state`` that is stored in its ``with_display_transforms``
member.

The ``perform_ondemand_calcs()`` step of the ``osg_compositor`` first calls
``traverse_display_requirements()`` then
``recstore_display_transforms::update()`` then waits for the transforms to
be complete via the ``wait_complete()`` method of the ``with_display_transforms``
member.

The ``perform_layer_rendering()`` step of the ``osg_compositor`` goes
through each channel to be rendered and performs the render according
to the ``renderer_type`` field of the ``display_requirement``, which
selects the image renderer (``SNDE_DRRT_IMAGE``), the geometry
renderer (``SNDE_DRRT_GEOMETRY``), or the waveform renderer
(``SNDE_DRRT_WAVEFORM``). The rendering is performed independently for
each channel into its own ``osg_layerwindow`` which wraps an
OpenGL "Frame Buffer Object" that stores the rendered output
for later compositing.

The renderers all access a shared ``osg_rendercache`` that manages
OpenSceneGraph scene graph elements that may be reusable. At the start
of the rendering pass, all elements in the RenderCache are marked
as ``potentially_obsolete``. Within the rendering pass the renderer
for each layer searches the cache for a previously rendered
scene graph entry rather than creating a new one. If one is found,
it clears the ``potentially_obsolete`` flag on that entry and
any recursive dependencies. Otherwise a new scene graph entry
is created and added to the cache. In many cases recursive
dependencies also need to be created, usually paralleling the
recursive structure of the ``display_requirement``. At the 
end of the rendering pass, all cache entries that are still
``potentially_obsolete`` are definitely no longer needed and
cleaned from the cache. 

The ``osg_rendercache`` finds the appropriate low-level renderer for a
``display_requirement`` from the ``rendermode`` generated by the
original recording display handler that created the
``display_requirement``. The render mode is a combination of a simple
mode STRING (usually ``SNDE_SRM_XXXX``) and the C++ type of the
original recording display handler. The appropriate low level renderer
is found by looking up that render mode in the
``osg_renderer_registry`` that is assembled during static initialization
of the program and as DLLs/shared objects are loaded.

Custom renderers can be added at run time by loading DLL(s) and shared
object(s) that call ``register_recording_display_handler()`` and/or
``osg_register_renderer()`` during static initialziation to add the
custom renderer components to their respective registries.
If the newly registered recording display handler is for a goal other than
``SNDE_SRG_RENDERING``, then it can be selected by setting the
``snde_render_goal`` metadata entry of the relevant recording to the
goal of the newly registered recording display handler. 

The entries in the ``osg_rendercache`` are indexed by the
``rendermode_ext`` which includes both the ``rendermode`` and a
``constraint`` field which should contains all of the parameters
specific to the rendering. This helps make sure that re-renders are
performed when needed.

For example, suppose the user selects a different colormap
scaling. The parameter of the on-demand colormapping math function
will change, triggering a recalculation.  In addition, the updated
parameter is part of the ``rendermode_ext`` constraint, so that the
renderer will rerender with the newly recalculated colormap output
rather than reusing the colormap scaling is selected by the user, the
constraint will be different, so the old rendercache entry will be
ignored and the image will be rerendered with the new colormap.
Since the ``potentially_obsolete`` flag of the old rendercache entry
is not cleared during the rendering pass, the old entry will
be discarded from the cache. 

Compositing Process
-------------------

The compositing process is split into several major phases: On-demand
calculations, rendering, and compositing.  The compositing process is
optionally split between multiple threads with different delegated
responsibilities. The process is basically a state machine with each
thread responsible for executing certain states.

The state management is handled through the ``next_state`` member of
the ``osg_compositor``. It is locked by the ``admin`` ``std::mutex``
of the ``osg_compositor`` and paired with the ``execution_notify``
condition variable. Responsibility mapping across threads is defined
in the ``responsibility_mapping`` map, which lists the various
responsibilities (``SNDE_OSGRCS_XXXX` defined in
``openscenegraph_compositor.hpp``) for each thread.

The need for various rerender operations is set via the
``need_rerender``, ``need_recomposite``, and ``need resize`` member
booleans. The ``dispatch()`` method is used by a thread to select its
action based on the thread characteristics (main thread needs to
return if idle vs. worker thread waits if idle), and the
responsibility mapping. The dispatch method looks at the next state
and executes that next state if it is the responsibility of the
calling thread. Otherwise it waits or returns as appropriate. If the
next state was ``SNDE_OSGRCS_WAITING`` it also looks at the member
booleans to see if a rerender or recomposite is needed and updates
``next_state`` if appropriate. ``dispatch()`` can also handle cleanup
requests and attempts to wake up the thread that will handle the
``next_state``.

In this fashion, the compositor executes on-demand calculations,
layer rendering, and compositing in sequence, optionally using
a separate thread either for the on-demand calculation or the
on-demand thread combined with layer rendering. All intermediate
outputs are cached so that only minimal work is done when
settings or inputs change. 


