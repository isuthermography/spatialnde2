

OBJS=allocator_test.o


all: allocator_test

clean:
	rm -f *~ allocator_test *.o *.bak

allocator_test: allocator_test.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CPPFLAGS)  -c $<


allocator_test.o: allocator_test.cpp

