import numpy as np

def quaternion_normalize(quat):
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion must have 4 components (real + 3 imaginary")
    
    norm = np.linalg.norm(quat, axis=-1)
    
    if np.isscalar(norm):
        if norm == 0:
            raise ValueError("Cannot normalize a zero quaternion")
        return quat/norm
    if (norm == 0).any():
        raise ValueError("Cannot normalize a zero quaternion")
    return quat/norm[..., np.newaxis]


def quaternion_product(quat1, quat2):
    if quat1.shape[-1] != 4:
        raise ValueError("Both quaternions must have 4 components (real + 3 imaginary).")
    if quat2.shape[-1] != 4:
        raise ValueError("Quaternion 2 must have 4 components (real + 3 imaginary).")
    
    # real = 0, i = 1, j = 2, k = 3
    
    # i*i = -1, j*j = -1, k*k = -1, i*j = k, j*k = i, k*i = j, j*i = -k, k*j = -i, i*k = -j
    # (q1[real] + q1[i]*i + q1[j]*j + q1[k]*k) * (q2[real] + q2[i]*i + q2[j]*j + q2[k]*k)
    # q1[real]*q2[real] + q1[real]*q2[i]*i + q1[real]*q2[j]*j + q1[real]*q2[k]*k + ...
    # q1[i]*i*q2[real] + q1[i]*i*q2[i]*i + q1[i]*i*q2[j]*j + q1[i]*i*q2[k]*k + ...
    # q1[j]*j*q2[real] + q1[j]*j*q2[i]*i + q1[j]*j*q2[j]*j + q1[j]*j*q2[k]*k + ...
    # q1[k]*k*q2[real] + q1[k]*k*q2[i]*i + q1[k]*k*q2[j]*j + q1[k]*k*q2[k]*k 

    # Compute the product using quaternion multiplication rules
    product = np.array([
        quat1[..., 0] * quat2[..., 0] - quat1[..., 1] * quat2[..., 1] - quat1[..., 2] * quat2[..., 2] - quat1[..., 3] * quat2[..., 3],  # Real part
        quat1[..., 0] * quat2[..., 1] + quat1[..., 1] * quat2[..., 0] + quat1[..., 2] * quat2[..., 3] - quat1[..., 3] * quat2[..., 2],  # i part
        quat1[..., 0] * quat2[..., 2] - quat1[..., 1] * quat2[..., 3] + quat1[..., 2] * quat2[..., 0] + quat1[..., 3] * quat2[..., 1],  # j part
        quat1[..., 0] * quat2[..., 3] + quat1[..., 1] * quat2[..., 2] - quat1[..., 2] * quat2[..., 1] + quat1[..., 3] * quat2[..., 0]   # k part
    ])
    return product

def quaternion_product_normalized(quat1, quat2): 
    unnormalized = quaternion_product(quat1, quat2)
    return quaternion_normalize(unnormalized)

def quaternion_inverse(quat):   
    norm = np.linalg.norm(quat, axis=-1)
    # norm = np.linalg.norm(quat)
    # if norm == 0:
    #     raise ValueError("Cannot compute the inverse of a zero quaternion")
    
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion must have 4 components (real + 3 imaginary")
    
    if np.isscalar(norm):
        if norm == 0:
            raise ValueError("Cannot normalize a zero quaternion")
        return quat/norm
    if (norm == 0).any():
        raise ValueError("Cannot normalize a zero quaternion")
    
    # The inverse of a quaternion [w, i, j, k] is [w, -i, -j, -k] normalized
    # inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]]) / (norm ** 2)
    inv = np.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)/(norm[..., np.newaxis]**2)
    return inv

def quaternion_apply_vector(quat, vec):
    if quat.shape[-1] != 4:
        raise ValueError("Quaternions must have 4 components (real + 3 imaginary)")
    if vec.shape[-1] != 3: 
        raise ValueError("Vectors must have 3 components")
    
    # vec_as_quat = np.array([0.0, *vec[:3]])
    vec_as_quat = np.concatenate([np.zeros((vec.shape[:-1], 1)), vec], axis=-1)
    
    q1_times_v = quaternion_product(quat, vec_as_quat)
    q1_inverse = quaternion_inverse(quat)
    applied_vector = quaternion_product(q1_times_v, q1_inverse)
    return applied_vector[..., 1:] # Return the vector without the real part

def quaternion_average(quat, axis=None):
    '''
    Quat is assumed to be a numpy array with the last axis representing the four 
    elements of a quaternion. Compute the average over a particular preceding axis
    or all preceding axes. 
    
    If axis is None, compute over all preceding axes. If axis is an integer, 
    compute over that axis. 
    '''
    # If compute over all axes, flatten them. 
    if axis is None: 
        quat = quat.reshape(np.prod(quat.shape[:-1]), 4)
        axis = 0
        pass
        
    # Generate the matrix of outer products of the quaternions over the 
    outer_prods = np.einsum("...i, ...j->...ij", quat, quat)
    
    # Evaluate eigenvectors/eigenvalues of outer product matrix
    (evals, evecs) = np.linalg.eigh(outer_prods)
    
    # Mean quaternion is the eigenvector corresponding to the largest (last) eigenvalue
    return evecs[..., :, -1]

def quaternion_apply_to_bothsides_of_matrix(quat, mtx): 
    '''
    Apply to LH side: Take the matrix (mtx), take each column, and apply to quat in the normal way. 
    Will give three transformed columns that can be reformed into a matrix. 
    Apply to the RH side: take the three rows from the matrix resulting from the previous step, treat them as columns, 
    apply to quat in the normal way. Will give three column vectors.
    take the three column vectors and interpret them as rows and that result is the transformed matrix.
    '''    
    # Normalize the quaternion 
    quat = quaternion_normalize(quat)
    
    # Apply quaternion to each column (LH side)
    # transformed_columns = np.array([quaternion_apply_vector(quat, mtx[:, i]) for i in range(mtx.shape[1])]) 
    transformed_columns = np.concatenate([quaternion_apply_vector(quat[..., np.newaxis, :], mtx[..., np.newaxis, :, i]) for i in range(mtx.shape[1])], axis=-2) 
    # Each row of transformed_columns represents a single transformed column
    
    # Apply quaternion across the transformed columns (RH side). Because they are stored as rows, we need to extract columns.
    # transformed_rows = np.array([quaternion_apply_vector(quat, transformed_columns[:, i]) for i in range(transformed_columns.shape[0])])
    transformed_rows = np.concatenate([quaternion_apply_vector(quat[..., np.newaxis, :], transformed_columns[..., np.newaxis, :, i]) for i in range(transformed_columns.shape[0])], axis=-2)

    # Each row of transformed_rows represents a single transformed row, which we now treat as a column vector. 
    # These column vectors per the original comment need to be interpreted as rows. 
    # Therefore, our transformed_mtx is just transformed_rows
    
    transformed_mtx = transformed_rows
    
    # # Normalize the quaternion 
    # quat = quaternion_normalize(quat)
    
    # # Build the rotation matrix from the quaternion
    # q_rot_mtx = quaternion_build_rotmtx(quat)

    # # Apply the rotation matrix to both sides of the original matrix
    # transformed_mtx = np.dot(np.dot(q_rot_mtx, mtx),q_rot_mtx.T)
    return transformed_mtx

def quaternion_build_rotmtx(quat):
    """ 
    Creates a 3x3 matrix with the upper 3x3 the rotation represented by quat. 
    The final column represents a zero offset. 
    """
    rotmtx = np.zeros(quat.shape[:-1] + [3, 3])
    vecs = [(1.0, 0.0, 0.0), 
                (0.0, 1.0, 0.0), 
                (0.0, 0.0, 1.0)]
    
    
    for i, vec in enumerate(vecs):
        rotmtx[..., :, i] = quaternion_apply_vector(quat, vec.reshape(*([1]*(len(quat.shape) - 1) + [3])))
    return rotmtx
