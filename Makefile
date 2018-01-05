# -Wno-ignored-attributes eliminates (apparently) spurious warnings about ignored attributes

CPPFLAGS=-g  -Wno-ignored-attributes


OBJS=allocator_test.o


all: allocator_test manager_test opencl_manager_test

clean:
	rm -f *~ allocator_test manager_test *.o *.bak opencl_manager_test

commit: clean
	hg addrem
	hg commit

allocator_test: allocator_test.o lockmanager.o allocator.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

manager_test: manager_test.o lockmanager.o 
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

opencl_manager_test: opencl_manager_test.o lockmanager.o openclcachemanager.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL

.cpp.o:
	$(CXX) $(CPPFLAGS)  -c $<


allocator_test.o: allocator_test.cpp
lockmanager.o: lockmanager.cpp
manager_test.o: manager_test.cpp
opencl_manager_test.o: opencl_manager_test.cpp
allocator.o: allocator.cpp
openclcachemanager.o: openclcachemanager.cpp
