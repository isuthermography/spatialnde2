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


static QUATERNION_INLINE void snde_null_orientation3(snde_orientation3 *out)
{
  snde_orientation3 null_orientation = { { { 0.0f, 0.0f, 0.0f, 0.0f } }, { {0.0f, 0.0f, 0.0f, 1.0f} } }; /* null offset vector and unit (null) quaternion */
  *out=null_orientation;
}

static QUATERNION_INLINE snde_bool quaternion_equal(const snde_coord4 a, const snde_coord4 b)
{
  return a.coord[0]==b.coord[0] && a.coord[1]==b.coord[1] && a.coord[2]==b.coord[2] && a.coord[3]==b.coord[3];
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


#ifndef __OPENCL_VERSION__
  assert(vec.coord[3]==0.0f);
#endif // __OPENCL_VERSION__
  
  snde_coord vnormsq = normsqvec3(&vec.coord[0]);
  
  quaternion_product(quat,vec,&q1_times_v);
  quaternion_inverse(quat,&q1_inverse);
  quaternion_product(q1_times_v,q1_inverse,product);

  // last coordinate of output should calculate to roughly 0
#ifndef __OPENCL_VERSION__
  assert(product->coord[3]*product->coord[3] <= 1e-13f*vnormsq);
#endif
  
  // ... and be written as exactly 0.
  product->coord[3]=0.0f;
}


static QUATERNION_INLINE void quaternion_build_rotmtx(const snde_coord4 quat,snde_coord4 *rotmtx /* (array of 3 or 4 coord4's, interpreted as column-major). Does not write 4th column  */ )
/* assumes quat is normalized, stored as 'i,j,k,w' components */
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
/* assumes quat is normalized, stored as 'i,j,k,w' components */
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
  orient->offset.coord[3]=0.0f; // always leave last element of offset as zero

  // Figure out quat
  // From https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
  // and based on Shuster, M. 1993, "A Survey of Attitude Representations", Journal of the Astronautical Sciences, 41(4):349-517  p 463-464

  // select equation according to which has the largest
  // square root paramter
  // eta4, Shuster eq. 163:
  snde_coord eta4_sqrt = 1.0 + rotmtx[0].coord[0] + rotmtx[1].coord[1] + rotmtx[2].coord[2];

  // eta1, Shuster eq. 166a
  snde_coord eta1_sqrt = 1.0 + rotmtx[0].coord[0] - rotmtx[1].coord[1] - rotmtx[2].coord[2];

  // eta2, Schuster eq 167a
  snde_coord eta2_sqrt = 1.0 - rotmtx[0].coord[0] + rotmtx[1].coord[1] - rotmtx[2].coord[2];

  // eta3, Schuster eq. 168a
  snde_coord eta3_sqrt = 1.0 - rotmtx[0].coord[0] -rotmtx[1].coord[1] + rotmtx[2].coord[2];

  // Note: eta1, eta2, eta3, eta4 represent quaternion components in order

  // NOTE: Compared with Shuster, we get the backwards rotation out (opposite sense) so we negate either the real part or the vector part to flip the sense
  
  if (eta4_sqrt >= eta1_sqrt && eta4_sqrt >= eta2_sqrt && eta4_sqrt >= eta3_sqrt) {
    // eta4_sqrt largest: Use eqs 163, 164
    
    orient->quat.coord[3]=0.5f*sqrt(eta4_sqrt); // eta4
    // In paper, matrix elements indexed (row, column) starting from 1
    // We index (column,row) starting from 0
    // vector part negated
    orient->quat.coord[0]=-(1.0f/(4.0f*orient->quat.coord[3]))*(rotmtx[2].coord[1]-rotmtx[1].coord[2]); // eta1

    orient->quat.coord[1]=-(1.0f/(4.0f*orient->quat.coord[3]))*(rotmtx[0].coord[2]-rotmtx[2].coord[0]); // eta2

    orient->quat.coord[2]=-(1.0f/(4.0f*orient->quat.coord[3]))*(rotmtx[1].coord[0]-rotmtx[0].coord[1]); // eta3
    
  } else if (eta1_sqrt >= eta3_sqrt && eta1_sqrt >= eta2_sqrt) {
    // eta1_sqrt largest: Use eqs 166
    assert(eta1_sqrt >= eta4_sqrt);

    orient->quat.coord[0] = 0.5f*sqrt(eta1_sqrt); // eta1
    orient->quat.coord[1] = (1.0f/(4.0f*orient->quat.coord[0]))*(rotmtx[1].coord[0] + rotmtx[0].coord[1]); // eta2
    orient->quat.coord[2] = (1.0f/(4.0f*orient->quat.coord[0]))*(rotmtx[2].coord[0] + rotmtx[0].coord[2]); // eta3
    // real part negated
    orient->quat.coord[3] = -(1.0f/(4.0f*orient->quat.coord[0]))*(rotmtx[2].coord[1]-rotmtx[1].coord[2]); // eta4
    
  } else if (eta2_sqrt > eta3_sqrt) {
    // eta2_sqrt largest: Use eqs 167
    assert(eta2_sqrt >= eta4_sqrt && eta2_sqrt >= eta1_sqrt);
    orient->quat.coord[1] = 0.5f*sqrt(eta2_sqrt); // eta2

    orient->quat.coord[0] = (1.0f/(4.0f*orient->quat.coord[1]))*(rotmtx[0].coord[1]+rotmtx[1].coord[0]); // eta1
    orient->quat.coord[2] = (1.0f/(4.0f*orient->quat.coord[1]))*(rotmtx[2].coord[1]+rotmtx[1].coord[2]); // eta3
    // real part negated
    orient->quat.coord[3] = -(1.0f/(4.0f*orient->quat.coord[1]))*(rotmtx[0].coord[2] - rotmtx[2].coord[0]); // eta4
    
  } else {
    // eta3_sqrt largest: Use eqs 168
    assert(eta3_sqrt >= eta4_sqrt && eta3_sqrt >= eta1_sqrt && eta3_sqrt >= eta2_sqrt);

    orient->quat.coord[2] = 0.5f*sqrt(eta3_sqrt); // eta3
    orient->quat.coord[0] = (1.0f/(4.0f*orient->quat.coord[2]))*(rotmtx[0].coord[2] + rotmtx[2].coord[0]); // eta1
    orient->quat.coord[1] = (1.0f/(4.0f*orient->quat.coord[2]))*(rotmtx[1].coord[2] + rotmtx[2].coord[1]); // eta2
    // real part negated
    orient->quat.coord[3] = -(1.0f/(4.0f*orient->quat.coord[2]))*(rotmtx[1].coord[0] - rotmtx[0].coord[1]); // eta4
    
    
  }
  // normalize the quaternion
  quaternion_normalize(orient->quat,&orient->quat);

  // Verify quat behavior by performing the inverse operation

  {
    snde_coord4 verify_rotmtx[4];
    unsigned row,col;
    snde_coord residual=0.0;
    
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
    assert(residual < 1e-4); // error residual should be minimal
  }
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
#ifndef __OPENCL_VERSION__
  assert(vec.coord[3] == 0.0f);
#endif
  
  quaternion_apply_vector(orient.quat,vec,out);
}

static QUATERNION_INLINE void orientation_apply_position(const snde_orientation3 orient,const snde_coord4 pos,snde_coord4 *out)
{
  /* for point p, q1pq1' + o1  */
  snde_coord4 posvec;
  snde_coord4 rotated_point;

#ifndef __OPENCL_VERSION__
  assert(pos.coord[3]==1.0f); // should be a position
#endif
  
  posvec=pos;
  posvec.coord[3]=0.0f;
  
  // rotate point
  quaternion_apply_vector(orient.quat,posvec,&rotated_point);

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
    
    quaternion_apply_vector(left.quat,right.offset,&rotated_right_offset);
    addcoordcoord4proj(rotated_right_offset,left.offset,&product->offset);
    
  }


#endif // SNDE_QUATERNION_H
