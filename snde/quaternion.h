#ifndef SNDE_QUATERNION_H
#define SNDE_QUATERNION_H


#ifndef __OPENCL_VERSION__

#include "snde/geometry_types.h"
#include "snde/vecops.h"

#endif // __OPENCL_VERSION__

#ifdef _MSC_VER
#define QUATERNION_INLINE  __inline
#else
#define QUATERNION_INLINE  inline
#endif

#define SNDE_QUAT_REAL 0
#define SNDE_QUAT_I 1
#define SNDE_QUAT_J 2
#define SNDE_QUAT_K 3

static QUATERNION_INLINE void snde_null_orientation3(snde_orientation3 *out)
{
  snde_orientation3 null_orientation = { { { 1.0f, 0.0f, 0.0f, 0.0f } }, { {0.0f, 0.0f, 0.0f, 1.0f} } }; /* null offset vector and unit (null) quaternion */
  *out=null_orientation;
}

static QUATERNION_INLINE void snde_invalid_orientation3(snde_orientation3 *out)
{
  snde_coord nan;
  nan = snde_infnan(0);
  
  out->offset.coord[0]=nan;
  out->offset.coord[1]=nan;
  out->offset.coord[2]=nan;
  out->offset.coord[3]=nan;
  out->quat.coord[0]=nan;
  out->quat.coord[1]=nan;
  out->quat.coord[2]=nan;
  out->quat.coord[3]=nan;
}

static QUATERNION_INLINE snde_bool quaternion_equal(const snde_coord4 a, const snde_coord4 b)
{
  return (a.coord[0]==b.coord[0] && a.coord[1]==b.coord[1] && a.coord[2]==b.coord[2] && a.coord[3]==b.coord[3]) || (a.coord[0]==-b.coord[0] && a.coord[1]==-b.coord[1] && a.coord[2]==-b.coord[2] && a.coord[3]==-b.coord[3]);
}

static QUATERNION_INLINE snde_bool orientation3_equal(const snde_orientation3 a, const snde_orientation3 b)
{
  return quaternion_equal(a.quat,b.quat) && equalcoord4(a.offset,b.offset);
}


static QUATERNION_INLINE void quaternion_normalize(const snde_coord4 unnormalized,snde_coord4 *normalized)
  /* returns the components of a normalized quaternion */
{
  snde_coord norm;
  
  norm=sqrt(pow(unnormalized.coord[0],2) + pow(unnormalized.coord[1],2) + pow(unnormalized.coord[2],2)+pow(unnormalized.coord[3],2));
  
  normalized->coord[0]=unnormalized.coord[0]/norm;
  normalized->coord[1]=unnormalized.coord[1]/norm;
  normalized->coord[2]=unnormalized.coord[2]/norm;
  normalized->coord[3]=unnormalized.coord[3]/norm;
  
}
  
static QUATERNION_INLINE void quaternion_product(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *product)
{
    /* quaternion coordinates are real part, i, j, k */
  product->coord[SNDE_QUAT_I]=quat1.coord[SNDE_QUAT_REAL]*quat2.coord[SNDE_QUAT_I] + quat1.coord[SNDE_QUAT_I]*quat2.coord[SNDE_QUAT_REAL] + quat1.coord[SNDE_QUAT_J]*quat2.coord[SNDE_QUAT_K] - quat1.coord[SNDE_QUAT_K]*quat2.coord[SNDE_QUAT_J];
  product->coord[SNDE_QUAT_J]=quat1.coord[SNDE_QUAT_REAL]*quat2.coord[SNDE_QUAT_J] + quat1.coord[SNDE_QUAT_J]*quat2.coord[SNDE_QUAT_REAL] - quat1.coord[SNDE_QUAT_I]*quat2.coord[SNDE_QUAT_K] + quat1.coord[SNDE_QUAT_K]*quat2.coord[SNDE_QUAT_I];
  product->coord[SNDE_QUAT_K]=quat1.coord[SNDE_QUAT_REAL]*quat2.coord[SNDE_QUAT_K] + quat1.coord[SNDE_QUAT_K]*quat2.coord[SNDE_QUAT_REAL] + quat1.coord[SNDE_QUAT_I]*quat2.coord[SNDE_QUAT_J] - quat1.coord[SNDE_QUAT_J]*quat2.coord[SNDE_QUAT_I];
  product->coord[SNDE_QUAT_REAL]=quat1.coord[SNDE_QUAT_REAL]*quat2.coord[SNDE_QUAT_REAL] - quat1.coord[SNDE_QUAT_I]*quat2.coord[SNDE_QUAT_I] - quat1.coord[SNDE_QUAT_J]*quat2.coord[SNDE_QUAT_J] - quat1.coord[SNDE_QUAT_K]*quat2.coord[SNDE_QUAT_K];
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

    snde_coord normsq;
  
    normsq=(pow(quat.coord[0],2) + pow(quat.coord[1],2) + pow(quat.coord[2],2)+pow(quat.coord[3],2));
  

    // quaternion inverse is the conjugate (the i,j,k terms negated) divided by the square of the magnitude.
    inverse->coord[0]=quat.coord[0]/normsq;
    inverse->coord[1]=-quat.coord[1]/normsq;
    inverse->coord[2]=-quat.coord[2]/normsq;
    inverse->coord[3]=-quat.coord[3]/normsq;
  }



static QUATERNION_INLINE void quaternion_apply_vector(const snde_coord4 quat,const snde_coord4 vec,snde_coord4 *product)
/* assumes quat is normalized, stored as 'w,i,j,k' components */
{
 


  // quaternion times vector
  //   = q1vq1'
  snde_coord4 q1_times_v;
  snde_coord4 q1_inverse;


#ifndef __OPENCL_VERSION__
  assert(vec.coord[3]==0.0f);
#endif // __OPENCL_VERSION__
  
  snde_coord vnormsq = normsqvec3(&vec.coord[0]);
  snde_coord4 quat_vec = {0.0,vec.coord[0],vec.coord[1],vec.coord[2]};
  
  quaternion_product(quat,quat_vec,&q1_times_v);
  quaternion_inverse(quat,&q1_inverse);
  quaternion_product(q1_times_v,q1_inverse,product);

  // real part of output should calculate to roughly 0
#ifndef __OPENCL_VERSION__
  assert(product->coord[SNDE_QUAT_REAL]*product->coord[SNDE_QUAT_REAL] <= 1e-13f*vnormsq);
#endif
  
  

  //move product from starting at the [1] element to starting at the [0] element
  product->coord[0]=product->coord[1];
  product->coord[1]=product->coord[2];
  product->coord[2]=product->coord[3];
  //result is a vector so last element is 0
  product->coord[3]=0.0f;
}

static QUATERNION_INLINE void quaternion_apply_position(const snde_coord4 quat,const snde_coord4  pos,snde_coord4 *product)
/* assumes quat is normalized, stored as 'i,j,k,w' components */
{
 

  snde_coord4 quat_vec;

  // quaternion times vector
  //   = q1vq1'
  snde_coord4 q1_times_v;
  snde_coord4 q1_inverse;


#ifndef __OPENCL_VERSION__
  assert(pos.coord[3]==1.0f);
#endif // __OPENCL_VERSION__
  quat_vec.coord[0]=0.0;
  quat_vec.coord[1]=pos.coord[0];
  quat_vec.coord[2]=pos.coord[1];
  quat_vec.coord[3]=pos.coord[2];
   
 
   
  
  snde_coord vnormsq = normsqvec3(&pos.coord[0]);
  
  quaternion_product(quat,quat_vec,&q1_times_v);
  quaternion_inverse(quat,&q1_inverse);
  quaternion_product(q1_times_v,q1_inverse,product);

  // real part of output should calculate to roughly 0
#ifndef __OPENCL_VERSION__
  assert(product->coord[0]*product->coord[0] <= 1e-13f*vnormsq);
#endif
  
  //move product from starting at the [1] element to starting at the [0] element
  product->coord[0]=product->coord[1];
  product->coord[1]=product->coord[2];
  product->coord[2]=product->coord[3];
  //result is a position so last element is 1
  product->coord[3]=1.0f;
}


static QUATERNION_INLINE void quaternion_build_rotmtx(const snde_coord4 quat,snde_coord4 *rotmtx /* (array of 3 or 4 coord4's, interpreted as column-major). Does not write 4th column  */ )
/*
  assumes quat is normalized, stored as 'w,i,j,k' components
  WARNING: when rotmtx is read out from python, it is a series
  of column vectors each of which is a snde_coord4. To get the
  actual matrix you need to extract the ["coord"] and then
  transpose, ie
  numpy_matrix = quaternion_build_rotmtx(...)["coord"].T
 */
  
{
  // This could definitely be optimized
  snde_coord4 vec1 = { { 1.0f, 0.0f, 0.0f, 0.0f } };
  quaternion_apply_vector(quat,vec1,&rotmtx[0]); // first column represents applying (1,0,0,0) vector

  snde_coord4 vec2 = { { 0.0f, 1.0f, 0.0f, 0.0f } };
  quaternion_apply_vector(quat,vec2,&rotmtx[1]); // second column represents applying (0,1,0,0) vector

  snde_coord4 vec3 = { { 0.0f, 0.0f, 1.0f, 0.0f } };
  quaternion_apply_vector(quat,vec3,&rotmtx[2]); // second column represents applying (0,0,1,0) vector

}

static QUATERNION_INLINE void orientation_build_rotmtx(const snde_orientation3 orient,snde_coord4 *rotmtx /* (array of 4 coord4's, interpreted as column-major).  */ )
/* assumes quat is normalized, stored as 'w,i,j,k' components */
{
  quaternion_build_rotmtx(orient.quat,rotmtx); // still need to do fourth column

  rotmtx[3] = orient.offset;
  rotmtx[3].coord[3]=1.0f; // lower right element of 4x4 always 1.0
}


static QUATERNION_INLINE void rotmtx_build_orientation(const snde_coord4 *rotmtx, // array of 4 coord4s, interpreted as column-major homogeneous coordinates 4x4
						       snde_orientation3 *orient)
{
  // offset is easy
  orient->offset = rotmtx[3];
  orient->offset.coord[3]=1.0f; // always leave last element of offset as 1.0

  // Figure out quat
  // From https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
  // and based on Shuster, M. 1993, "A Survey of Attitude Representations", Journal of the Astronautical Sciences, 41(4):349-517  p 463-464

  // select equation according to which has the largest
  // square root paramter
  // eta4, Shuster eq. 163:
  snde_coord eta4_sqrt = 1.0f + rotmtx[0].coord[0] + rotmtx[1].coord[1] + rotmtx[2].coord[2];

  // eta1, Shuster eq. 166a
  snde_coord eta1_sqrt = 1.0f + rotmtx[0].coord[0] - rotmtx[1].coord[1] - rotmtx[2].coord[2];

  // eta2, Schuster eq 167a
  snde_coord eta2_sqrt = 1.0f - rotmtx[0].coord[0] + rotmtx[1].coord[1] - rotmtx[2].coord[2];

  // eta3, Schuster eq. 168a
  snde_coord eta3_sqrt = 1.0f - rotmtx[0].coord[0] -rotmtx[1].coord[1] + rotmtx[2].coord[2];

  // Note: eta1, eta2, eta3, eta4 represent quaternion components in order

  // NOTE: Compared with Shuster, we get the backwards rotation out (opposite sense) so we negate either the real part or the vector part to flip the sense
  
  if (eta4_sqrt >= eta1_sqrt && eta4_sqrt >= eta2_sqrt && eta4_sqrt >= eta3_sqrt) {
    // eta4_sqrt largest: Use eqs 163, 164
    
    orient->quat.coord[SNDE_QUAT_REAL]=0.5f*sqrt(eta4_sqrt); // eta4
    // In paper, matrix elements indexed (row, column) starting from 1
    // We index (column,row) starting from 0
    // vector part negated
    orient->quat.coord[SNDE_QUAT_I]=-(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_REAL]))*(rotmtx[2].coord[1]-rotmtx[1].coord[2]); // eta1

    orient->quat.coord[SNDE_QUAT_J]=-(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_REAL]))*(rotmtx[0].coord[2]-rotmtx[2].coord[0]); // eta2

    orient->quat.coord[SNDE_QUAT_K]=-(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_REAL]))*(rotmtx[1].coord[0]-rotmtx[0].coord[1]); // eta3
    
  } else if (eta1_sqrt >= eta3_sqrt && eta1_sqrt >= eta2_sqrt) {
    // eta1_sqrt largest: Use eqs 166
#ifndef __OPENCL_VERSION__
    assert(eta1_sqrt >= eta4_sqrt);
#endif
    
    orient->quat.coord[SNDE_QUAT_I] = 0.5f*sqrt(eta1_sqrt); // eta1
    orient->quat.coord[SNDE_QUAT_J] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_I]))*(rotmtx[1].coord[0] + rotmtx[0].coord[1]); // eta2
    orient->quat.coord[SNDE_QUAT_K] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_I]))*(rotmtx[2].coord[0] + rotmtx[0].coord[2]); // eta3
    // real part negated
    orient->quat.coord[SNDE_QUAT_REAL] = -(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_I]))*(rotmtx[2].coord[1]-rotmtx[1].coord[2]); // eta4
    
  } else if (eta2_sqrt > eta3_sqrt) {
    // eta2_sqrt largest: Use eqs 167
#ifndef __OPENCL_VERSION__
    assert(eta2_sqrt >= eta4_sqrt && eta2_sqrt >= eta1_sqrt);
#endif
    orient->quat.coord[SNDE_QUAT_J] = 0.5f*sqrt(eta2_sqrt); // eta2

    orient->quat.coord[SNDE_QUAT_I] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_J]))*(rotmtx[0].coord[1]+rotmtx[1].coord[0]); // eta1
    orient->quat.coord[SNDE_QUAT_K] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_J]))*(rotmtx[2].coord[1]+rotmtx[1].coord[2]); // eta3
    // real part negated
    orient->quat.coord[SNDE_QUAT_REAL] = -(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_J]))*(rotmtx[0].coord[2] - rotmtx[2].coord[0]); // eta4
    
  } else {
    // eta3_sqrt largest: Use eqs 168
#ifndef __OPENCL_VERSION__
    assert(eta3_sqrt >= eta4_sqrt && eta3_sqrt >= eta1_sqrt && eta3_sqrt >= eta2_sqrt);
#endif

    orient->quat.coord[SNDE_QUAT_K] = 0.5f*sqrt(eta3_sqrt); // eta3
    orient->quat.coord[SNDE_QUAT_I] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_K]))*(rotmtx[0].coord[2] + rotmtx[2].coord[0]); // eta1
    orient->quat.coord[SNDE_QUAT_J] = (1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_K]))*(rotmtx[1].coord[2] + rotmtx[2].coord[1]); // eta2
    // real part negated
    orient->quat.coord[SNDE_QUAT_REAL] = -(1.0f/(4.0f*orient->quat.coord[SNDE_QUAT_K]))*(rotmtx[1].coord[0] - rotmtx[0].coord[1]); // eta4
    
    
  }
  // normalize the quaternion
  quaternion_normalize(orient->quat,&orient->quat);

  // Verify quat behavior by performing the inverse operation

  {
    snde_coord4 verify_rotmtx[4];
    unsigned row,col;
    snde_coord residual=0.0f;
    
    orientation_build_rotmtx(*orient,verify_rotmtx);
    // check match of upper 3x3
    for (col=0;col < 3;col++) {
      for (row=0; row < 3; row++) {
	residual += pow(verify_rotmtx[col].coord[row]-rotmtx[col].coord[row],2.0f);
      }
    }
    // !!!**** If this triggers and the residual is large
    // there is a good change your upper 3x3 is not a rotation
    // but a reflection. Check its determinant and if negative
    // then it is a reflection. 
#ifndef __OPENCL_VERSION__
    assert(residual < 1e-4); // error residual should be minimal
#endif
  }
}

static QUATERNION_INLINE int orientation_valid(const snde_orientation3 orient)
{
  if (isnan(orient.offset.coord[0]) || isnan(orient.offset.coord[1]) || isnan(orient.offset.coord[2]) || isnan(orient.offset.coord[3]) || isnan(orient.quat.coord[0]) || isnan(orient.quat.coord[1]) || isnan(orient.quat.coord[2]) || isnan(orient.quat.coord[3])) {
    return FALSE;
  }
  snde_coord norm;
  
  norm=sqrt(pow(orient.quat.coord[0],2) + pow(orient.quat.coord[1],2) + pow(orient.quat.coord[2],2)+pow(orient.quat.coord[3],2));
  
  if (fabs(norm-1.0) > 1e-3) {
    // quaternion is way out of normalization
    return FALSE;
  }

  if (fabs(orient.offset.coord[3]-1.0) > 1e-3) {
    // last element of offset is way off 1.0
    return FALSE;
  }
    
  
  return TRUE;
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
  if (isnan(orient.quat.coord[0])) {
    snde_invalid_orientation3(inverse);
    return;
  }
  
  quaternion_inverse(orient.quat,&inverse->quat);
  quaternion_apply_position(inverse->quat,orient.offset,&inverse->offset);
  inverse->offset.coord[0]=-inverse->offset.coord[0];
  inverse->offset.coord[1]=-inverse->offset.coord[1];
  inverse->offset.coord[2]=-inverse->offset.coord[2];
  
}

static QUATERNION_INLINE void orientation_apply_vector(const snde_orientation3 orient,const snde_coord4 vec,snde_coord4 *out)
{
#ifndef __OPENCL_VERSION__
  assert(vec.coord[3] == 0.0f);
#endif
  
  quaternion_apply_vector(orient.quat,vec,out);
}

static QUATERNION_INLINE void orientation_apply_position(const snde_orientation3 orient,const snde_coord4 pos,snde_coord4 *out)
{
  /* for point p, q1pq1' + o1  */
  //  snde_coord4 posvec;
  snde_coord4 rotated_point;

#ifndef __OPENCL_VERSION__
  assert(pos.coord[3]==1.0f); // should be a position
#endif
  
  // posvec=pos;
  // posvec.coord[3]=0.0f;
  
  // rotate point
  quaternion_apply_position(orient.quat,pos,&rotated_point);

  // add offset
  addcoordcoord4proj(rotated_point,orient.offset,out);
  out->coord[3]=1.0f; // a position
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
    
    quaternion_apply_position(left.quat,right.offset,&rotated_right_offset);
    addcoordcoord4proj(rotated_right_offset,left.offset,&product->offset);
    product->offset.coord[3]=1.0;
  }


#endif // SNDE_QUATERNION_H
