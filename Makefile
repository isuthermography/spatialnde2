# -Wno-ignored-attributes eliminates (apparently) spurious warnings about ignored attributes

CPPFLAGS=-g  -Wno-ignored-attributes -DSNDE_LOCKMANAGER_COROUTINES_THREADED


OBJS=allocator_test.o


all: allocator_test manager_test opencl_manager_test

clean:
	rm -f *~ allocator_test manager_test *.o *.bak opencl_manager_test *_c.h *_h.h

commit: clean
	hg addrem
	hg commit

allocator_test: allocator_test.o lockmanager.o allocator.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

manager_test: manager_test.o lockmanager.o 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

opencl_manager_test: opencl_manager_test.o lockmanager.o openclcachemanager.o opencl_utils.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL -lpthread

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
