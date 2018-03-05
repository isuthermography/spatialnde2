import numpy as np

translation = [6.2, 4.3, 6.9]
center = [1.0, 2.0, 3.3]
rotation = [8.0, 2.0, 2.6, 9.8]
scaleOrientation = [3.0, 3.0, 2.2, 2.5]
scale = [2.0, 1.2, 3.5]

T = np.matrix(((1.0, 0.0, 0.0, translation[0]),
               (0.0, 1.0, 0.0, translation[1]),
               (0.0, 0.0, 1.0, translation[2]),
               (0.0, 0.0, 0.0, 1.0)), dtype='d')
C = np.matrix(((1.0, 0.0, 0.0, center[0]),
               (0.0, 1.0, 0.0, center[1]),
               (0.0, 0.0, 1.0, center[2]),
               (0.0, 0.0, 0.0, 1.0)), dtype='d')

# Apply Rodrigues rotation formula to determine R and SR
k = rotation[:3]
ang = rotation[3]
kmag = np.linalg.norm(k)
if kmag == 0.0:
    kmag = 1.0  # null rotation
    k = np.array((0.0, 0.0, 1.0), dtype='d')
    ang = 0.0
    pass
k /= kmag

# cross product matrix
RK = np.matrix(((0.0, -k[2], k[1]),
                (k[2], 0.0, -k[0]),
                (-k[1], k[0], 0.0)), dtype='d')
# R=np.eye(3) + np.sin(ang)*RK + (1.0-np.cos(ang))*np.dot(RK,RK)

R = np.concatenate(
    (np.concatenate((np.eye(3) + np.sin(ang) * RK + (1.0 - np.cos(ang)) * np.dot(RK, RK), np.zeros((3, 1), dtype='d'))
                    , axis=1),
     np.array(((0.0, 0.0, 0.0, 1.0),), dtype='d')), axis=0)

# Apply Rodrigues rotation formula to determine scale orientation
SOk = scaleOrientation[:3]

SOang = scaleOrientation[3]
SOkmag = np.linalg.norm(SOk)
if SOkmag == 0.0:
    SOkmag = 1.0  # null rotation
    SOk = np.array((0.0, 0.0, 1.0), dtype='d')
    SOang = 0.0
    pass
SOk /= SOkmag

# cross product matrix
SOK = np.matrix(((0.0, -SOk[2], SOk[1]),
                 (SOk[2], 0.0, -SOk[0]),
                 (-SOk[1], SOk[0], 0.0)), dtype='d')
SR = np.concatenate((np.concatenate(
    (np.eye(3) + np.sin(SOang) * SOK + (1.0 - np.cos(SOang)) * np.dot(SOK, SOK), np.zeros((3, 1), dtype='d')), axis=1),
                     np.array(((0.0, 0.0, 0.0, 1.0),), dtype='d')), axis=0)

S = np.matrix(((scale[0], 0.0, 0.0, 0.0),
               (0.0, scale[1], 0.0, 0.0),
               (0.0, 0.0, scale[2], 0.0),
               (0.0, 0.0, 0.0, 1.0)), dtype='d')

# Transform components are defined as matrices not arrays,
# so we can just multiply them

eval_array = np.array((T * C * R * SR * S * (-SR) * (-C)), dtype='d')

print(eval_array)
