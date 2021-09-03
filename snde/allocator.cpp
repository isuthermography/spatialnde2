
#include <cstdint>

#include "snde/geometry_types.h"
#include "snde/allocator.h"


extern "C" {

  // BUG... these implementations should probably catch exceptions.... 
  /*
  snde_index snde_allocator_alloc(snde::allocator *alloc,snde_index nelem)
  {
    return alloc->alloc(nelem);
  }
  
  void snde_allocator_free(snde::allocator *alloc,snde_index addr,snde_index nelem)
  {
    alloc->free(addr,nelem);
  }
  */  
};

namespace snde {
  std::map<unsigned,unsigned> prime_factorization(unsigned number)
  {
    // return prime factorization as vector of (integer,power) pairs
    unsigned divisor=2;
    unsigned power=0;
    
    std::map<unsigned,unsigned> factors_powers;
    
    while (number >= 2) {
      /* evaluate possible factors up to sqrt(value), 
	 dividing them out as we find them */
      if ((number % divisor)==0) {
	power++;
	if (power==1) {
	  //factors[reqnum].push_back(divisor);
	  factors_powers.emplace(std::make_pair(divisor,power));
	} else {
	  factors_powers.at(divisor) = power;
	}
	number = number / divisor;
	
	
	
      } else {
	assert((number % divisor)!=0);
	power=0;
	divisor++;
      }
      
      assert(divisor <= number || number==1);
    }
    
    
    return factors_powers;
  }
}

namespace snde {
  unsigned multiply_factors(std::map<unsigned,unsigned> factors)
  {
    unsigned product=1;
    unsigned cnt; 
    for (auto && factor_power: factors) {
      for (cnt=0; cnt < factor_power.second;cnt++) {
	product *= factor_power.first;
      }
    }
    return product; 
  }
}
