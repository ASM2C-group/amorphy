import numpy as np
import sys
import itertools
from inputValues import A, B, C, minkowski_reduce_cell
from numba import njit, int64, jit


@njit
def numpy_norm_axis_0(a):
    norms = np.empty(a.shape[1], dtype=a.dtype)
    for i in range(a.shape[1]):
        s = 0.0
        for j in range(a.shape[0]):
            s += a[j,i]**2
        norms[i] = np.sqrt(s)
    return norms

@njit
def numpy_norm_axis_1(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in range(a.shape[0]):
        s = 0.0
        for j in range(a.shape[1]):
            s += a[i,j]**2
        norms[i] = np.sqrt(s)
    return norms

def numpy_dot(cs, joint):
    # print(cs.shape, joint.shape)
    assert cs.shape[1] == joint.shape[0]
    res = [[0.0000 for x in range(cs.shape[1])] for y in range(cs.shape[0])]
    for i in range(len(cs)):        
        for j in range(len(joint[0])):
            for k in range(len(joint)):
               res[i][j] += cs[i][k] * joint[k][j]
    return res

@njit
def wrap_positions(positions, cell):
    # shift = np.asarray(center) - 0.5 - eps

    # Don't change coordinates when pbc is False
    # shift[np.logical_not(pbc)] = 0.0
    # assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    shift = np.zeros(3)
    fractional = np.linalg.solve(cell.T, np.asarray(positions).T).T - shift

    for i in range(3):        
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]
    return np.dot(fractional, cell)



def general_find_mic(v):
    #rcell, _ = minkowski_reduce(cell)
    rcell = minkowski_reduce_cell
    positions = wrap_positions(v, rcell)
    # ranges = [np.arange(-1 * p, p + 1) for p in pbc]
    ranges = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])

    hkls = np.array([(0, 0, 0)] + list(itertools.product(*ranges)))
    vrvecs = hkls @ rcell

    # Map positions into neighbouring cells.
    x = positions + vrvecs[:, None]

    # Find minimum images
    lengths = np.linalg.norm(x, axis=2)
    indices = np.argmin(lengths, axis=0)
    vmin = x[indices, np.arange(len(positions)), :]
    vlen = lengths[indices, np.arange(len(positions))]
    return vmin, vlen

@njit
def naive_find_mic(v, cell):
    f = np.linalg.solve(cell.T, np.transpose(v)).T
    f -= np.floor(f + 0.5)
    vmin = f @ cell
    vlen = np.linalg.norm(vmin)
    return vmin, vlen


def find_mic(v, cell):
    v = np.atleast_2d(v)
    naive_find_mic_is_safe = False
    vmin, vlen = naive_find_mic(v, cell)
    # naive find mic is safe only for the following condition
    if (vlen < 0.5 * min(A, B, C)).all():
        naive_find_mic_is_safe = True  # hence skip Minkowski reduction
    if not naive_find_mic_is_safe:
        vmin, vlen = general_find_mic(v)
    return vmin, vlen


def get_distance(p1, p2, cell):
    cell = np.array(cell, dtype=float)
    vector = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    D, D_len = find_mic(vector, cell=cell)
    return D , D_len


def get_angle(v0, v1, cell):
    """ Calculate anlges in degrees between vector v0 and v1"""
    cell = np.array(cell, dtype=float)
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    v0, nv0 = find_mic(v0, cell=cell)
    v1, nv1 = find_mic(v1, cell=cell)

    if (nv0 <= 0) or (nv1 <= 0):
        raise ZeroDivisionError('Undefined angle')
    v0n = v0 / nv0
    v1n = v1 / nv1
    # We just normalized the vectors, but in some cases we can get
    # bad things like 1+2e-16.  These we clip away:

    angle = np.arccos(np.einsum('ij,ij->i', v0n, v1n).clip(-1.0, 1.0))
    return np.degrees(angle[0])
