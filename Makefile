# -Wno-ignored-attributes eliminates (apparently) spurious warnings about ignored attributes

CPPFLAGS=-g -Wall -fPIC -Wno-int-in-bool-context -Wno-ignored-attributes -DSNDE_LOCKMANAGER_COROUTINES_THREADED -I/usr/include/eigen3 -I/usr/include/libxml2


OBJS=allocator_test.o

LDFLAGS=-lpthread

all: allocator_test manager_test opencl_manager_test x3d_test

swig: geometry_types_h.h lockmanager.o openclcachemanager.o opencl_utils.o
	swig -c++ -python spatialnde2.i
	g++ -I/usr/include/python2.7 $(CPPFLAGS) -c spatialnde2_wrap.cxx
	g++ -shared spatialnde2_wrap.o lockmanager.o openclcachemanager.o opencl_utils.o -o _spatialnde2.so -lOpenCL -lpthread

clean:
	rm -f *~ allocator_test manager_test *.o *.bak opencl_manager_test *_c.h *_h.h x3d_test _spatialnde2.so _spatialnde2.dll spatialnde2_wrap.cxx *.pyc

commit: clean
	hg addrem
	hg commit

allocator_test: allocator_test.o lockmanager.o allocator.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

manager_test: manager_test.o lockmanager.o 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) 

opencl_manager_test: opencl_manager_test.o lockmanager.o openclcachemanager.o opencl_utils.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL -lpthread

x3d_test: x3d_test.o 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lxml2 

.cpp.o:
	$(CXX) $(CPPFLAGS)  -c $<

%_c.h: %.c
	python file2header.py $< $@

%_h.h: %.h
	python file2header.py $< $@


allocator_test.o: allocator_test.cpp
lockmanager.o: lockmanager.cpp
manager_test.o: manager_test.cpp
opencl_manager_test.o: opencl_manager_test.cpp testkernel_c.h geometry_types_h.h
allocator.o: allocator.cpp
openclcachemanager.o: openclcachemanager.cpp
opencl_utils.o: opencl_utils.cpp
testkernel_c.h: testkernel.c
geometry_types_h.h: geometry_types.h
x3d_test.o: x3d_test.cpp x3d.hpp
