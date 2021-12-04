#ifndef SNDE_QUATERNION_H
#define SNDE_QUATERNION_H

#include "snde/geometry_types.h"
#include "snde/vecops.h"

#ifdef _MSC_VER
#define QUATERNION_INLINE  __inline
#else
#define QUATERNION_INLINE  inline
#endif


static QUATERNION_INLINE void snde_null_orientation3(snde_orientation3 *out)
{
  snde_orientation3 null_orientation = { { 0.0, 0.0, 0.0, 0.0 }, {0.0, 0.0, 0.0, 1.0} }; /* null offset vector and unit (null) quaternion */
  *out=null_orientation;
}

static QUATERNION_INLINE void quaternion_normalize(const snde_coord4 unnormalized,snde_coord4 *normalized)
  /* returns the components of a normalized quaternion */
  {
    double norm;
    
    norm=sqrt(pow(unnormalized.coord[0],2) + pow(unnormalized.coord[1],2) + pow(unnormalized.coord[2],2)+pow(unnormalized.coord[3],2));

    normalized->coord[0]=unnormalized.coord[0]/norm;
    normalized->coord[1]=unnormalized.coord[1]/norm;
    normalized->coord[2]=unnormalized.coord[2]/norm;
    normalized->coord[3]=unnormalized.coord[3]/norm;

  }
  
static QUATERNION_INLINE void quaternion_product(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *product)
{
    /* quaternion coordinates are i, j, k, real part */
  product->coord[0]=quat1.coord[3]*quat2.coord[0] + quat1.coord[0]*quat2.coord[3] + quat1.coord[1]*quat2.coord[2] - quat1.coord[2]*quat2.coord[1];
  product->coord[1]=quat1.coord[3]*quat2.coord[1] + quat1.coord[1]*quat2.coord[3] - quat1.coord[0]*quat2.coord[2] + quat1.coord[2]*quat2.coord[0];
  product->coord[2]=quat1.coord[3]*quat2.coord[2] + quat1.coord[2]*quat2.coord[3] + quat1.coord[0]*quat2.coord[1] - quat1.coord[1]*quat2.coord[0];
  product->coord[3]=quat1.coord[3]*quat2.coord[3] - quat1.coord[0]*quat2.coord[0] - quat1.coord[1]*quat2.coord[1] - quat1.coord[2]*quat2.coord[2];
  
}


static QUATERNION_INLINE void quaternion_product_normalized(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *product)
{
  snde_coord4 unnormalized;

  quaternion_product(quat1,quat2,&unnormalized);
  
  quaternion_normalize(unnormalized,product);
}

static QUATERNION_INLINE void quaternion_inverse(const snde_coord4 quat, snde_coord4 *inverse)
  {
    /* quaternion coordinates are i, j, k, real part */

    quaternion_normalize(quat,inverse);

    // quaternion inverse is normalized with the i,j,k terms negated
    inverse->coord[0]=-inverse->coord[0];
    inverse->coord[1]=-inverse->coord[1];
    inverse->coord[2]=-inverse->coord[2];
  }



static QUATERNION_INLINE void quaternion_apply_vector(const snde_coord4 quat,const snde_coord4 vec,snde_coord4 *product)
/* assumes quat is normalized, stored as 'i,j,k,w' components */
{
  //snde_coord matrix[9];
  
  ///* first row */
  //matrix[0]=pow(quat.coord[0],2)-pow(quat.coord[1],2)-pow(quat.coord[2],2)+pow(quat.coord[3],2);
  //matrix[1]=2.0*(quat.coord[0]*quat.coord[1] - quat.coord[3]*quat.coord[2]);
  //matrix[2]=2.0*(quat.coord[0]*quat.coord[2] + quat.coord[3]*quat.coord[1]);
  ///* second row */
  //matrix[3]=2.0*(quat.coord[0]*quat.coord[1] + quat.coord[3]*quat.coord[2]);
  //matrix[4]=-pow(quat.coord[0],2) + pow(quat.coord[1],2) - pow(quat.coord[2],2) + pow(quat.coord[3],2);
  //matrix[5]=2.0*(quat.coord[1]*quat.coord[2] - quat.coord[3]*quat.coord[0]);
  ///* third row */
  //matrix[6]=2.0*(quat.coord[0]*quat.coord[2] - quat.coord[3]*quat.coord[1]);
  //matrix[7]=2.0*(quat.coord[1]*quat.coord[2] + quat.coord[3]*quat.coord[0]);
  //matrix[8]=-pow(quat.coord[0],2) - pow(quat.coord[1],2) + pow(quat.coord[2],2) + pow(quat.coord[3],2);
  //
  //unsigned rowcnt,colcnt;
  //
  //for (rowcnt=0;rowcnt < 3; rowcnt++) {
  //  product->coord[rowcnt]=0;
  //  for (colcnt=0;colcnt < 3; colcnt++) {
  //    product->coord[rowcnt] += matrix[rowcnt*3 + colcnt] * vec.coord[colcnt];
  //  }
  //}


  // quaternion times vector
  //   = q1vq1'
  snde_coord4 q1_times_v;
  snde_coord4 q1_inverse;

  assert(vec.coord[3]==0.0);
  
  quaternion_product(quat,vec,&q1_times_v);
  quaternion_inverse(quat,&q1_inverse);
  quaternion_product(q1_times_v,q1_inverse,product);

  assert(product->coord[3]==0.0);
}


static QUATERNION_INLINE void quaternion_build_rotmtx(const snde_coord4 quat,snde_coord4 *rotmtx /* (array of 3 or 4 coord4's, interpreted as column-major). Does not write 4th column  */ )
/* assumes quat is normalized, stored as 'i,j,k,w' components */
{
  // This could definitely be optimized
  snde_coord4 vec1 = { 1.0, 0.0, 0.0, 0.0};
  quaternion_apply_vector(quat,vec1,&rotmtx[0]); // first column represents applying (1,0,0,0) vector

  snde_coord4 vec2 = { 0.0, 1.0, 0.0, 0.0};
  quaternion_apply_vector(quat,vec2,&rotmtx[1]); // second column represents applying (0,1,0,0) vector

  snde_coord4 vec3 = { 0.0, 0.0, 1.0, 0.0};
  quaternion_apply_vector(quat,vec3,&rotmtx[2]); // second column represents applying (0,0,1,0) vector

}

static QUATERNION_INLINE void orientation_build_rotmtx(const snde_orientation3 orient,snde_coord4 *rotmtx /* (array of 4 coord4's, interpreted as column-major).  */ )
/* assumes quat is normalized, stored as 'i,j,k,w' components */
{
  quaternion_build_rotmtx(orient.quat,rotmtx); // still need to do fourth column

  rotmtx[3] = orient.offset;
  
}


static QUATERNION_INLINE void orientation_inverse(const snde_orientation3 orient,snde_orientation3 *inverse)
{
  // point p, rotated by the orientation q1, o1 is
  // p_rot = q1pq1' + o1
  //   ... solve for p
  // q1'p_rotq1 = q1'q1pq1'q1 + q1'o1q1
  // q1'p_rotq1 = p + q1'o1q1
  // p = q1'p_rotq1 - q1'o1q1
  // Therefore, the orientation inverse
  // is q1', -q1'o1q1
  quaternion_inverse(orient.quat,&inverse->quat);
  quaternion_apply_vector(inverse->quat,orient.offset,&inverse->offset);
  inverse->offset.coord[0]=-inverse->offset.coord[0];
  inverse->offset.coord[1]=-inverse->offset.coord[1];
  inverse->offset.coord[2]=-inverse->offset.coord[2];
  
}

static QUATERNION_INLINE void orientation_apply_vector(const snde_orientation3 orient,const snde_coord4 vec,snde_coord4 *out)
{
  assert(vec.coord[3] == 0.0);
  quaternion_apply_vector(orient.quat,vec,out);
}

static QUATERNION_INLINE void orientation_apply_position(const snde_orientation3 orient,const snde_coord4 pos,snde_coord4 *out)
{
  /* for point p, q1pq1' + o1  */
  snde_coord4 posvec;
  snde_coord4 rotated_point;

  assert(pos.coord[3]==1.0); // should be a position
  
  posvec=pos;
  posvec.coord[3]=0.0;
  
  // rotate point
  quaternion_apply_vector(orient.quat,posvec,&rotated_point);

  // add offset
  addcoordcoord4proj(rotated_point,orient.offset,out);
  out->coord[3]=1.0; // a position
}

static QUATERNION_INLINE void orientation_orientation_multiply(const snde_orientation3 left,const snde_orientation3 right,snde_orientation3 *product)
  {
      /* orientation_orientation_multiply must consider both quaternion and offset **/
      /* for vector v, quat rotation is q1vq1' */
      /* for point p, q1pq1' + o1  */
      /* for vector v double rotation is q2q1vq1'q2' ... where q2=left, q1=right */
      /* for point p  q2(q1pq1' + o1)q2' + o2 */
      /*             = q2q1pq1'q2' + q2o1q2' + o2 */
      /* so given q2, q1,   and o2, o1
	 product quaternion is q2q1
         product offset is q2o1q2' + o2 */
    snde_coord4 rotated_right_offset;

    quaternion_product_normalized(left.quat,right.quat,&product->quat);
    
    quaternion_apply_vector(left.quat,right.offset,&rotated_right_offset);
    addcoordcoord4proj(rotated_right_offset,left.offset,&product->offset);
    
  }


#endif // SNDE_QUATERNION_H
