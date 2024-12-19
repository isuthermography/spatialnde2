# Test quatpy.py
import numpy as np
import quatpy as q

def test_quaternion_normalize():
    # Test for non-zero quaternion
    quat = np.array([[1, 0, 0, 0]])  # Identity quaternion
    normalized = q.quaternion_normalize(quat)
    assert np.allclose(np.linalg.norm(normalized, axis=-1), 1.0), "Normalized quaternion should have unit norm."
    
    # Test for zero quaternion
    quat_zero = np.array([[0, 0, 0, 0]])
    try:
        q.quaternion_normalize(quat_zero)
        assert False, "Expected ValueError for zero quaternion"
    except ValueError as e:
        assert str(e) == "Cannot normalize a zero quaternion", f"Unexpected error message: {e}"

def test_quaternion_product():
    # Test valid quaternions
    
    quat1 = np.array([1, 2, 3, 4])
    quat2 = np.array([5, 6, 7, 8])
    product = q.quaternion_product(quat1, quat2)
    # print(f"q1 {quat1}")
    # print(f"q2 {quat2}")
    # print(f"product {product}")
    expected_product = np.array([-60, 12, 30, 24])
    

    assert np.allclose(product, expected_product), "Quaternion product is not correct."
    assert product.shape == (4,), f"Expected product shape (4) got {product.shape}"
    
    quat3 = np.array([[1, 2, 3, 4]]) # double brackets indicate a leading axis
    quat4 = np.array([[5, 6, 7, 8]])
    product2 = q.quaternion_product(quat3, quat4)
    assert np.allclose(product2, expected_product)

    # Test for invalid quaternion shapes, will fail if code is working correctly (commented out to run other tests)
    # quat_invalid = np.array([[1, 0, 0]])  # 3 components instead of 4
    # invalid_test = q.quaternion_product(quat1, quat_invalid)

def test_quaternion_product_normalized():
    quat1 = np.array([[1, 0, 0, 0]])
    quat2 = np.array([[0, 1, 0, 0]])
    product_normalized = q.quaternion_product_normalized(quat1, quat2)
    # print(f'product_noramalized {product_normalized}')
    assert product_normalized.shape == (1, 4), f"Expected product shape (1, 4), got {product_normalized.shape}"
    assert np.allclose(np.linalg.norm(product_normalized, axis=-1), 1.0), "Normalized product quaternion should have unit norm."

def test_quaternion_inverse():
    quat = np.array([[1, 0, 0, 0]])
    inverse = q.quaternion_inverse(quat)
    # print(f"quat {quat}")
    assert np.allclose(inverse, quat), "Inverse of identity quaternion should be itself."
    assert inverse.shape == (1, 4), f"Expected product shape (1, 4), got {inverse.shape}" # make sure shape is still 1,4

    quat2 = np.array([[1, 2, 3, 4]])
    inverse2 = q.quaternion_inverse(quat2)
    # print(f"inverse2 {inverse2}")
    expected_inverse = np.array([[1, -2, -3, -4]]) / np.linalg.norm(quat2) ** 2
    # print(f"expected_inverse{expected_inverse}")
    assert np.allclose(inverse2, expected_inverse), "Quaternion inverse is not correct."
    assert inverse2.shape == (1, 4), f"Expected product shape (1, 4), got {inverse2.shape}" # make sure shape is still 1,4
    
    quat_zero = np.array([[0, 0, 0, 0]])
    try:
        q.quaternion_inverse(quat_zero)
        assert False, "Expected ValueError for zero quaternion"
    except ValueError as e:
        assert str(e) == "Cannot normalize a zero quaternion", f"Unexpected error message: {e}"

def test_quaternion_apply_vector():
    quat = np.array([[1, 0, 0, 0]])  # Identity quaternion
    vec = np.array([[1, 0, 0]])  # Vector to apply the quaternion to
    applied_vec = q.quaternion_apply_vector(quat, vec)
    assert np.allclose(applied_vec, vec), "Identity quaternion should not alter the vector."

    quat_rot = np.array([[0.7071, 0, 0.7071, 0]])  # 90 degrees rotation quaternion around y-axis
    vec_rot = np.array([[1, 0, 0]])  # Should rotate to [0, 0, -1]
    applied_vec_rot = q.quaternion_apply_vector(quat_rot, vec_rot)
    assert np.allclose(applied_vec_rot, np.array([[0, 0, -1]])), "Quaternion rotation applied incorrectly."

    # Testing leading axes
    # Comparing to this
    quat = np.array([1, 2, 3, 4])  # Identity quaternion
    vec = np.array([5, 6, 7])  # Vector to apply the quaternion to
    applied_vec2 = q.quaternion_apply_vector(quat, vec)
    
    # Provide a random quat with a leading axis and a vector with corresponding leading axis and they should give the same result
    quat1 = np.array([1,2,3,4])
    quat1_axis = quat1[np.newaxis,:]
    vec1 = np.array([5,6,7])
    vec_axis = vec1[np.newaxis,:]
    applied_axis = q.quaternion_apply_vector(quat1_axis,vec_axis)
    assert np.allclose(applied_axis, applied_vec2), "Applied vectors do not match"

def test_quaternion_average():
    # quats = np.array([[[1, 1, 1, 1], [-1, -1, -1, -1]]]) # 1 is not normalized, 0.5 is normalized, should output 0.5
    quats = np.array([[[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]]])
    # quat1 = np.array([1, 2, 4, 8]) 
    # quat2 = np.array([4, 3, 5, 2]) 
    quat1 = q.quaternion_normalize(np.random.rand(4))
    quat2 = q.quaternion_normalize(np.random.rand(4))
    q_avg = q.quaternion_average(quats)
    # print(f"avg_quat {q_avg}")
    
    # angle between average and each one should be smaller than between each one 
    angle = q.angle_between_quaternions(quat1, quat2)
    # print(f"angle {np.degrees(angle)}")
    angle_average = q.angle_between_quaternions(quat1, q_avg)
    # print(f"angle_average {np.degrees(angle_average)}")
    # assert angle_average <= angle, f"Average angle between q1 and q_avg is greater than the original angle"
        
    expected_avg_quat = np.array([[ 0.5, 0.5, 0.5, 0.5]])  # Need correct value
    assert np.allclose(q_avg, expected_avg_quat), f"Expected {expected_avg_quat}, but got {q_avg}"

def test_quaternion_apply_to_bothsides_of_matrix():
    # Test 1: Identity quaternion
    quat = np.array([4, 3, 2, 1])/np.linalg.norm([4,3,2,1])  # Identity quaternion
    mtx = np.eye(3)  # Identity matrix
    transformed_mtx = q.quaternion_apply_to_bothsides_of_matrix(quat, mtx)
    # print(f"transformed_mtx = \n{transformed_mtx}")
    assert np.allclose(transformed_mtx, mtx), "Applying identity quaternion should return the same matrix."
    
    # Generate a random quaternion, normalize it, 
    # generate random 3x3 matrix, generate a random quat, normalize it, use the apply to both sides to transform it 
    # generate the rotation matrix equivalent of the quat and apply to both sides of random mat and should get same thing (build_rot_mtx)
    # mtx2 = np.random.rand(3, 3)
    # print(f"{mtx2}")
    
    # What does this matrix need to be? This is required for apply_to_bothsides, in LTA script we use the I matrix
    # symmetric 3x3 matrix
    symmetric_matx = np.array([
        [2, 4, 6],
        [4, 5, 7],
        [6, 7, 8]
    ])
    # print("Symmetric 3x3 Matrix:")
    quat2 = np.random.rand(4)  
    quat2 = q.quaternion_normalize(quat2)
    # print(f"quat2{quat2}")
    build_rotmtx  = q.quaternion_build_rotmtx(quat2)
    transform = q.quaternion_apply_to_bothsides_of_matrix(quat2, symmetric_matx)
    # print(f"quaternion_apply_to_bothsides_of_matrix\n {transform}")

    # print(f"quaternion_build_rotmtx \n {build_rotmtx}")
    build_rotmtx_applied = build_rotmtx@symmetric_matx@build_rotmtx.transpose()
    assert np.allclose(transform, build_rotmtx_applied ), "quaternion_apply_to_bothsides_of_matrix does not match quaternion_build_rotmtx"


def test_angle_between_quaternion():
    q1 = np.array([1, 2, 3, 4])  # Identity quaternion (no rotation)
    q2 = np.array([5, 6, 7, 8])  # 90-degree rotation around the x-axis
    
    angle_radians = q.angle_between_quaternions(q1, q2)
    
    angle_degrees = np.degrees(angle_radians) # Convert the angle to degrees
    
    print(f"angle_degrees {angle_degrees}")

# def run_tests():
#     print("Running tests...")
#     print("Running test_quaternion_normalize")
#     test_quaternion_normalize() # PASSED
#     print("Running test_quaternion_product")
    # test_quaternion_product() # PASSED
    # print("Running test_quaternion_product_normalized")
#     test_quaternion_product_normalized() # PASSED
#     print("Running test_quaternion_inverse")
#     test_quaternion_inverse() # PASSED
#     print("Running test_quaternion_apply_vector")
#     test_quaternion_apply_vector() # dimension issue
#     # print("Running test_quaternion_average") # SOMETIMES PASSES, Angle is issue
#     # test_quaternion_average() # PASSED
    # print("Running test_quaternion_apply_to_bothsides_of_matrix")
    # test_quaternion_apply_to_bothsides_of_matrix()
#     # test_angle_between_quaternion()
    
def run_tests():
    print("Running tests...")

    # # Helper function to print with colors
    def print_result(test_name, success, message=""):
        if success:
            print(f"\033[92m{test_name} PASSED\033[0m")  # Green for success
        else:
            print(f"\033[91m{test_name} FAILED: {message}\033[0m")  # Red for failure

    # Test for quaternion normalization
    try:
        print("Running test_quaternion_normalize")
        test_quaternion_normalize()  # Assuming this test raises an error on failure
        print_result("test_quaternion_normalize", True)
    except Exception as e:
        print_result("test_quaternion_normalize", False, str(e))

    # Test for quaternion product
    try:
        print("Running test_quaternion_product")
        test_quaternion_product()  # Assuming this test raises an error on failure
        print_result("test_quaternion_product", True)
    except Exception as e:
        print_result("test_quaternion_product", False, str(e))

    # Test for quaternion product normalized
    try:
        print("Running test_quaternion_product_normalized")
        test_quaternion_product_normalized()  # Assuming this test raises an error on failure
        print_result("test_quaternion_product_normalized", True)
    except Exception as e:
        print_result("test_quaternion_product_normalized", False, str(e))

    # Test for quaternion inverse
    try:
        print("Running test_quaternion_inverse")
        test_quaternion_inverse()  # Assuming this test raises an error on failure
        print_result("test_quaternion_inverse", True)
    except Exception as e:
        print_result("test_quaternion_inverse", False, str(e))

    # Test for quaternion apply to vector
    try:
        print("Running test_quaternion_apply_vector")
        test_quaternion_apply_vector()  # Assuming this test raises an error on failure
        print_result("test_quaternion_apply_vector", True)
    except Exception as e:
        print_result("test_quaternion_apply_vector", False, str(e))

    # Test for quaternion average
    try:
        print("Running test_quaternion_average")
        test_quaternion_average()  # Assuming this test raises an error on failure
        print_result("test_quaternion_average", True)
    except Exception as e:
        print_result("test_quaternion_average", False, str(e))

    # Test for quaternion apply to both sides of a matrix
    try:
        print("Running test_quaternion_apply_to_bothsides_of_matrix")
        test_quaternion_apply_to_bothsides_of_matrix()  # Assuming this test raises an error on failure
        print_result("test_quaternion_apply_to_bothsides_of_matrix", True)
    except Exception as e:
        print_result("test_quaternion_apply_to_bothsides_of_matrix", False, str(e))

    print("All tests complete.")

run_tests()
