import numpy as np
def forward_kinematics_ur3e(q):
    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[0])
    TransRotZ[0, 1] = -np.sin(q[0])
    TransRotZ[1, 0] = np.sin(q[0])
    TransRotZ[1, 1] = np.cos(q[0])
    TransRotZ[2, 3] = 0.15185
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(np.pi / 2)
    TransRotX[1, 2] = -np.sin(np.pi / 2)
    TransRotX[2, 1] = np.sin(np.pi / 2)
    TransRotX[2, 2] = np.cos(np.pi / 2)
    TransRotX[0, 3] = 0.
    T_0_1 = TransRotZ @ TransRotX

    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[1])
    TransRotZ[0, 1] = -np.sin(q[1])
    TransRotZ[1, 0] = np.sin(q[1])
    TransRotZ[1, 1] = np.cos(q[1])
    TransRotZ[2, 3] = 0.
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(0.)
    TransRotX[1, 2] = -np.sin(0.)
    TransRotX[2, 1] = np.sin(0.)
    TransRotX[2, 2] = np.cos(0.)
    TransRotX[0, 3] = -0.24355
    T_1_2 = TransRotZ @ TransRotX

    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[2])
    TransRotZ[0, 1] = -np.sin(q[2])
    TransRotZ[1, 0] = np.sin(q[2])
    TransRotZ[1, 1] = np.cos(q[2])
    TransRotZ[2, 3] = 0.
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(0.)
    TransRotX[1, 2] = -np.sin(0.)
    TransRotX[2, 1] = np.sin(0.)
    TransRotX[2, 2] = np.cos(0.)
    TransRotX[0, 3] = -0.2132
    T_2_3 = TransRotZ @ TransRotX

    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[3])
    TransRotZ[0, 1] = -np.sin(q[3])
    TransRotZ[1, 0] = np.sin(q[3])
    TransRotZ[1, 1] = np.cos(q[3])
    TransRotZ[2, 3] = 0.13105
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(np.pi / 2)
    TransRotX[1, 2] = -np.sin(np.pi / 2)
    TransRotX[2, 1] = np.sin(np.pi / 2)
    TransRotX[2, 2] = np.cos(np.pi / 2)
    TransRotX[0, 3] = 0.
    T_3_4 = TransRotZ @ TransRotX

    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[4])
    TransRotZ[0, 1] = -np.sin(q[4])
    TransRotZ[1, 0] = np.sin(q[4])
    TransRotZ[1, 1] = np.cos(q[4])
    TransRotZ[2, 3] = 0.08535
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(-np.pi / 2)
    TransRotX[1, 2] = -np.sin(-np.pi / 2)
    TransRotX[2, 1] = np.sin(-np.pi / 2)
    TransRotX[2, 2] = np.cos(-np.pi / 2)
    TransRotX[0, 3] = 0.
    T_4_5 = TransRotZ @ TransRotX

    TransRotZ = np.identity(4)
    TransRotZ[0, 0] = np.cos(q[5])
    TransRotZ[0, 1] = -np.sin(q[5])
    TransRotZ[1, 0] = np.sin(q[5])
    TransRotZ[1, 1] = np.cos(q[5])
    TransRotZ[2, 3] = 0.0921
    TransRotX = np.identity(4)
    TransRotX[1, 1] = np.cos(0.)
    TransRotX[1, 2] = -np.sin(0.)
    TransRotX[2, 1] = np.sin(0.)
    TransRotX[2, 2] = np.cos(0.)
    TransRotX[0, 3] = 0.
    T_5_6 = TransRotZ @ TransRotX

    X_B_TCP = T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6
    return X_B_TCP