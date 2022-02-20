%{

  #include "snde/quaternion.h"
  
%}

void snde_null_orientation3(snde_orientation3 *OUTPUT);

snde_bool quaternion_equal(const snde_coord4 a, const snde_coord4 b);

snde_bool orientation3_equal(const snde_orientation3 a, const snde_orientation3 b);


void quaternion_normalize(const snde_coord4 unnormalized,snde_coord4 *OUTPUT);
  
void quaternion_product(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *OUTPUT);


void quaternion_product_normalized(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *OUTPUT);

void quaternion_inverse(const snde_coord4 quat, snde_coord4 *OUTPUT);


void quaternion_apply_vector(const snde_coord4 quat,const snde_coord4 vec,snde_coord4 *OUTPUT);


void quaternion_build_rotmtx(const snde_coord4 quat,snde_coord4 *OUTPUT /* (array of 3 or 4 coord4's, interpreted as column-major). Does not write 4th column  */ );

void orientation_build_rotmtx(const snde_orientation3 orient,snde_coord4 *OUTPUT /* (array of 4 coord4's, interpreted as column-major).  */ );


void orientation_inverse(const snde_orientation3 orient,snde_orientation3 *OUTPUT);

void orientation_apply_vector(const snde_orientation3 orient,const snde_coord4 vec,snde_coord4 *OUTPUT);

void orientation_apply_position(const snde_orientation3 orient,const snde_coord4 pos,snde_coord4 *OUTPUT);

void orientation_orientation_multiply(const snde_orientation3 left,const snde_orientation3 right,snde_orientation3 *OUTPUT);
