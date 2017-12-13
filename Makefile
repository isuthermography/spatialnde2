CPPFLAGS=-g

OBJS=allocator_test.o


all: allocator_test manager_test

clean:
	rm -f *~ allocator_test manager_test *.o *.bak

commit: clean
	hg addrem
	hg commit

allocator_test: allocator_test.o lockmanager.o allocator.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

manager_test: manager_test.o lockmanager.o arraymanager.o
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CPPFLAGS)  -c $<


allocator_test.o: allocator_test.cpp
lockmanager.o: lockmanager.cpp
manager_test.o: manager_test.cpp
allocator.o: allocator.cpp
arraymanager.o: arraymanager.cpp
