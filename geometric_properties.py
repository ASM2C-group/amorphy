import numpy as np
from numpy import pi, sin, cos, arccos, sqrt, dot
from numpy.linalg import norm

eps = 10e-9

def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y / norm(y)

def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y) / (norm(x) * norm(y))) * 180. / pi


def cellMatrix_to_cellParameter(cell, radians=False):
    """Returns the cell parameters [a, b, c, alpha, beta, gamma].

    Angles are in degrees unless radian=True is used.
    """
    lengths = [np.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / pi * arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * pi / 180 for angle in angles]
    return np.array(lengths + angles)


def cellParameter_to_cellMatrix(cellpar, ab_normal=(0, 0, 1), a_direction=None):
    """Return a 3x3 cell matrix from cellpar=[a,b,c,alpha,beta,gamma].

    Angles must be in degrees.

    The returned cell is orientated such that a and b
    are normal to `ab_normal` and a is parallel to the projection of
    `a_direction` in the a-b plane.

    Default `a_direction` is (1,0,0), unless this is parallel to
    `ab_normal`, in which case default `a_direction` is (0,0,1).

    The returned cell has the vectors va, vb and vc along the rows. The
    cell will be oriented such that va and vb are normal to `ab_normal`
    and va will be along the projection of `a_direction` onto the a-b
    plane.

    Example:

    >>> cell = cellpar_to_cell([1, 2, 4, 10, 20, 30], (0, 1, 1), (1, 2, 3))
    >>> np.round(cell, 3)
    array([[ 0.816, -0.408,  0.408],
           [ 1.992, -0.13 ,  0.13 ],
           [ 3.859, -0.745,  0.745]])

    """
    if a_direction is None:
        if np.linalg.norm(np.cross(ab_normal, (1, 0, 0))) < 1e-5:
            a_direction = (0, 0, 1)
        else:
            a_direction = (1, 0, 0)

    # Define rotated X,Y,Z-system, with Z along ab_normal and X along
    # the projection of a_direction onto the normal plane of Z.
    ad = np.array(a_direction)
    Z = unit_vector(ab_normal)
    X = unit_vector(ad - dot(ad, Z) * Z)
    Y = np.cross(Z, X)

    # Express va, vb and vc in the X,Y,Z-system
    alpha, beta, gamma = 90., 90., 90.
    if isinstance(cellpar, (int, float)):
        a = b = c = cellpar
    elif len(cellpar) == 1:
        a = b = c = cellpar[0]
    elif len(cellpar) == 3:
        a, b, c = cellpar
    else:
        a, b, c, alpha, beta, gamma = cellpar

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    # alpha
    if abs(abs(alpha) - 90) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = cos(alpha * pi / 180.0)
    # beta
    if abs(abs(beta) - 90) < eps:
        cos_beta = 0.0
    else:
        cos_beta = cos(beta * pi / 180.0)
    # gamma
    if abs(gamma - 90) < eps:
        cos_gamma = 0.0
        sin_gamma = 1.0
    elif abs(gamma + 90) < eps:
        cos_gamma = 0.0
        sin_gamma = -1.0
    else:
        cos_gamma = cos(gamma * pi / 180.0)
        sin_gamma = sin(gamma * pi / 180.0)

    # Build the cell vectors
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cx * cx - cy * cy
    assert cz_sqr >= 0
    cz = sqrt(cz_sqr)
    vc = c * np.array([cx, cy, cz])

    # Convert to the Cartesian x,y,z-system
    abc = np.vstack((va, vb, vc))
    T = np.vstack((X, Y, Z))
    cell = dot(abc, T)

    return cell


def volume_sphere(r, A, B, C):
    volume = 4.0 / 3.0 * np.pi * r**3    #  volume of sphere

    if r > min(A,B,C) /2.0 :             # this will remove the spherical cap if we go more than defined.
        volume -= 6 * np.pi * (( 2 * r +  (min(A,B,C) / 2.0) ) * ( r - ( min(A,B,C) / 2.0 ))**2)/ 3
    return volume

def check_orthorhombic(cell):

    ''' np.flatnonzero : This flattens the 3D matrix into 1D matrix and gives
        the indices of non-zero elements. If the cell is orthorhombic, all the
        indices of diagonal element will be printed by this function.
        Moreover %4 gives the remainder of those indices which infact is 0
        for orthorhombic system and thus later .any() checks for presence
        of any non-zero element.
        Thus if orthorhombic the below function returns True
    '''

    Decesion =  not (np.flatnonzero(cell) % 4).any()
    return Decesion

def rotate(angle, around_vector, position, center=(0,0,0), radian=False):

    '''This function gives the rotates xyz coordinates of position by an angle of 
       "angle (in deg)" along vector "around_vector" with reference to "center" 
       point.

       Parameters:
       ------------------------------------------------------------------------
       angle         : Angle to be rotated in degrees
       around_vector : Vector along the rotation has to be performed
       position      : xyz coordinates of point to be rotated
       center        : Coordinate along which rotation has to be performed along 
                       "around_vector".
       radian        : Boolean (True/False). If true, "angle" is in radians.
       
       Return:
       ------------------------------------------------------------------------
       position      : Rotated (counter-clockwise) xyz coordinates 

    '''
    around_vector = np.array(around_vector, dtype=float)

    norm = np.linalg.norm
    normv = norm(around_vector)
    if normv == 0.0:
        raise ZeroDivisionError('Cannot rotate: norm(around_vector) == 0')

    if not radian:
        angle *= np.pi / 180

    around_vector /= normv
    c = np.cos(angle)
    s = np.sin(angle)

    center = np.asarray(center, float)
    position = np.asarray(position, float)

    p = position - center

    position[:] = (c * p - np.cross(p, s * around_vector) +
                   np.outer(np.dot(p, around_vector), (1.0 - c) * around_vector) + 
                   center)

    position[np.abs(position) < eps] = 0
    return position


