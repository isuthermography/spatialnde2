%shared_ptr(snde::memallocator);
%shared_ptr(snde::cmemallocator);

%{
  
#include "memallocator.hpp"
%}


namespace snde {
  class memallocator {
  public:
    virtual void *malloc(std::size_t nbytes)=0;
    virtual void *calloc(std::size_t nbytes)=0;
    virtual void *realloc(void *ptr,std::size_t newsize)=0;
    virtual void free(void *ptr)=0;
    virtual ~memallocator()  {};

  };

  class cmemallocator: public memallocator {
  public:
    void *malloc(std::size_t nbytes);
  
    void *calloc(std::size_t nbytes);

    void *realloc(void *ptr,std::size_t newsize);

    void free(void *ptr);

    ~cmemallocator();
  };
}

