import numpy as np
from inputValues import A, B, C, LatticeMatrix, is_orthorhombic
from numpy.linalg import norm
from distance_minkowski_reduction import get_distance, get_angle
from numba import njit, jit

@jit
def displacement(a, b):

    if not is_orthorhombic:
        disp, dist = get_distance(a, b, cell=LatticeMatrix)

        return disp, dist
    
    else:
        r_vec = np.zeros(3)

        dx =  (a[0] - b[0])
        abs_x = abs(dx)
        if abs_x > A : abs_x = abs_x % A
        if abs_x > A/2 and dx > 0   : x = -min(abs_x, A-abs_x)
        elif abs_x > A/2 and dx < 0 : x = min(abs_x, A-abs_x)
        elif abs_x <= A/2 and dx < 0: x = -abs_x
        elif abs_x <= A/2 and dx > 0: x = abs_x
        else: x = dx
        r_vec[0] = x

        dy =  (a[1] - b[1])
        abs_y = abs(dy)
        if abs_y > B : abs_y = abs_y % B
        if abs_y > B/2 and dy > 0   : y = -min(abs_y, B-abs_y)
        elif abs_y > B/2 and dy < 0 : y = min(abs_y, B-abs_y)
        elif abs_y <= B/2 and dy < 0: y = -abs_y
        elif abs_y <= B/2 and dy > 0: y = abs_y
        else: y = dy
        r_vec[1] = y

        dz =  (a[2] - b[2])
        abs_z = abs(dz)
        if abs_z > C : abs_z = abs_z % C
        if abs_z > C/2 and dz > 0   : z = -min(abs_z, C-abs_z)
        elif abs_z > C/2 and dz < 0 : z = min(abs_z, C-abs_z)
        elif abs_z <= C/2 and dz < 0: z = -abs_z
        elif abs_z <= C/2 and dz > 0: z = abs_z
        else: z = dz
        r_vec[2] = z    

        return r_vec, norm(r_vec)

def angle(a, b, c):
    '''Here b is assumed to be the coordinates of central atom'''

    if not is_orthorhombic:
        v12 = a - b
        v32 = c - b
        angle = get_angle(v12, v32, cell=LatticeMatrix)
        return angle
    
    else:
        v12, v12_len = displacement(a, b)
        v32, v32_len = displacement(c, b)
        
        v12n = v12 / v12_len
        v32n = v32 / v32_len
        
        # We just normalized the vectors, but in some cases we can get
        # bad things like 1+2e-16.  These we clip away:
        angle = np.arccos(np.dot(v12n, v32n).clip(-1.0, 1.0))
        return np.degrees(angle)
