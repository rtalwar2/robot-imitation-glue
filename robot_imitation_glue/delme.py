import numpy as np

from airo_spatial_algebra.se3 import SE3Container
pose =np.array([[-0.99908236,  0.04283037,  0.      ,   -0.51505265], [ 0.04283037,  0.99908236  ,0.     ,     0.01400662], [ 0.,    0.,   -1.,    0.16756164],[ 0.,    0.,    0.,    1.        ]])
p_B_TCP_touch = [0,0,0]   # the position where the TCP will touch your 3D point
R_B_TCP_touch_X = np.array([1,0,0])  # rotation of TCP around X-axis  
R_B_TCP_touch_Y = np.array([0,-1,0])  # rotation of TCP around Y-axis  
R_B_TCP_touch_Z = np.array([0,0,-1])  # rotation of TCP around Z-axis  

X_B_TCP_touch_se3 = SE3Container.from_orthogonal_base_vectors_and_translation(
    R_B_TCP_touch_X, R_B_TCP_touch_Y, R_B_TCP_touch_Z, p_B_TCP_touch
)
X_B_TCP_touch = X_B_TCP_touch_se3.homogeneous_matrix
# get angle between two rotation matrices
R1 = pose[:3, :3]
R2 = X_B_TCP_touch[:3, :3]
R_diff = R1.T @ R2
angle = np.arccos((np.trace(R_diff) - 1) / 2)
print("angle between R1 and R2:", np.degrees(angle), "degrees")

