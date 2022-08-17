import numpy as np
import itertools

TOL = 1E-12
MAX_IT = 100000    # in practice this is not exceeded


class CycleChecker:

    def __init__(self, d):
        #assert d in [2, 3]

        # worst case is the hexagonal cell in 2D and the fcc cell in 3D
        n = {2: 6, 3: 12}[d]
        
        # max cycle length is total number of primtive cell descriptions
        max_cycle_length = np.prod([n - i for i in range(d)]) * np.prod(d)
        self.visited = np.zeros((max_cycle_length, 3 * d), dtype=int)

    def add_site(self, H):
        # flatten array for simplicity
        H = H.ravel()

        # check if site exists
        found = (self.visited == H).all(axis=1).any()

        # shift all visited sites down and place current site at the top
        self.visited = np.roll(self.visited, 1, axis=0)
        self.visited[0] = H
        return found



def relevant_vectors_2D(u, v):
    cs = np.array([e for e in itertools.product([-1, 0, 1], repeat=2)])
    vs = cs @ [u, v]
    indices = np.argsort(np.linalg.norm(vs, axis=1))[:7]
    return vs[indices], cs[indices]

def closest_vector(t0, u, v):
    t = t0

    a = np.zeros(2, dtype=int)
    
    rs, cs = relevant_vectors_2D(u, v)
    dprev = float("inf")

    for _ in range(MAX_IT):
        ds = np.linalg.norm(rs + t, axis=1)
        index = np.argmin(ds)
        if index == 0 or ds[index] >= dprev:
            return a

        dprev = ds[index]
        r = rs[index]
        kopt = int(round(-np.dot(t, r) / np.dot(r, r)))
        a += kopt * cs[index]
        t = t0 + a[0] * u + a[1] * v

    raise RuntimeError(f"Closest vector not found after {MAX_IT} iterations")


def reduction_gauss(B, hu, hv):
    """Calculate a Gauss-reduced lattice basis (2D reduction)."""
    cycle_checker = CycleChecker(d=2)
    u = hu @ B
    v = hv @ B

    for _ in range(MAX_IT):
        x = int(round(np.dot(u, v) / np.dot(u, u)))
        hu, hv = hv - x * hu, hu
        u = hu @ B
        v = hv @ B
        site = np.array([hu, hv])
        if np.dot(u, u) >= np.dot(v, v) or cycle_checker.add_site(site):
            return hv, hu

    raise RuntimeError(f"Gaussian basis not found after {MAX_IT} iterations")

def reduction_full(BD):
    """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
    cycle_checker = CycleChecker(d=3)
    H = np.eye(3, dtype=int)
    norms = np.linalg.norm(BD, axis=1)

    for it in range(MAX_IT):
        # Sort vectors by norm
        H = H[np.argsort(norms, kind='merge')]

        # Gauss-reduce smallest two vectors
        hw = H[2]
        hu, hv = reduction_gauss(BD, H[0], H[1])
        H = np.array([hu, hv, hw])
        R = H @ BD

        # Orthogonalize vectors using Gram-Schmidt
        u, v, _ = R
        X = u / np.linalg.norm(u)
        Y = v - X * np.dot(v, X)
        Y /= np.linalg.norm(Y)

        # Find closest vector to last element of R
        pu, pv, pw = R @ np.array([X, Y]).T
        nb = closest_vector(pw, pu, pv)

        # Update basis
        H[2] = [nb[0], nb[1], 1] @ H
        R = H @ BD

        norms = np.linalg.norm(R, axis=1)
        if norms[2] >= norms[1] or cycle_checker.add_site(H):
            return R, H

    raise RuntimeError(f"Reduced basis not found after {MAX_IT} iterations")

def handedness(cell) -> int:
    """Sign of the determinant of the matrix of cell vectors.

    1 for right-handed cells, -1 for left, and 0 for cells that
    do not span three dimensions."""
    return int(np.sign(np.linalg.det(cell)))


def minkowski_reduce(cell):
    cell = np.array(cell, dtype=float)
    op = np.eye(3, dtype=int)
    _, op = reduction_full(cell)

    # maintain cell handedness
    if handedness(cell) != handedness(op @ cell):
        op = -op

    norms1 = np.sort(np.linalg.norm(cell, axis=1))
    norms2 = np.sort(np.linalg.norm(op @ cell, axis=1))
    if (norms2 > norms1 + TOL).any():
        raise RuntimeError("Minkowski reduction failed")
    return op @ cell

