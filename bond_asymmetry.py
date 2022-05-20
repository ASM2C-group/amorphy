from numba import njit
from periodic_boundary_condition import displacement
import numpy as np

#@njit
def BondAsymmetry(coordination, Central_atom, Bonding_atom):
    
    ''' Here the funtion f_i, calculates the amount of asymmetry for a host atom with its bonding atoms.
        f_i = 2/(m(m-1))* sum_{j=1}^{m-1} * sum_{k=j+1}^{m} * (d_ij - d_ik) 
        m = Total number of bonding atom.
        Variable Central atom and Bonding atom contains the coordinates.

        Parameters:
        ---------------------------------------------------------------
        coordination : coordination number
        Central_atom : coordination of central atom
        Bonding_atom : [list] Coordinates of bonding atoms

        Return:
        ---------------------------------------------------------------
        funtion f_i

    '''
    # print(coordination, Central_atom, len(Bonding_atom))
    
    if coordination == 0 : 
        return np.nan  # No bonding atoms found
    elif coordination == 1 : 
        return 0       # No asymmetry with single bonded atoms
    
    constant = 2 / ( coordination * ( coordination - 1) )
    
    dist_diff = 0
    
    for j in range(coordination-1):
        _, dist_a = displacement(Central_atom, Bonding_atom[j])
        for k in range(j+1, coordination):
            _, dist_b = displacement(Central_atom, Bonding_atom[k])
            dist_diff += abs(dist_a - dist_b)
    
    return constant * dist_diff
