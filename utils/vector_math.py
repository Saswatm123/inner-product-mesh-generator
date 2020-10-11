import numpy as np
from numpy.linalg import norm, inv
from utils.general import arrayargs, NdarrayHashKey

@arrayargs
def vproj_vector(v1, v2):
    '''
        Args:
            v1:
                Vector to project onto v2
            v2:
                Vector to use as base for projection
        Desc:
            Returns v1 projected onto v2
    '''
    return (v1 @ (v2/(norm(v2)**2) ) ) * v2

@arrayargs
def vproj_basis(v, basis):
    '''
        Args:
            v:
                Vector to project onto basis
            basis:
                Basis of hyperplane to project v onto. Does not assume basis is orthogonal
        Desc:
            Returns v projected onto basis. Since no orthogonality assumption, we use
            iterative projection method
    '''

    proj = np.zeros(v.shape)
    for b in basis:
        vprojb = vproj_vector(v, b)
        v = v - vprojb
        proj += vprojb
    return proj

@arrayargs
def vdiff_basis(v, basis):
    '''
        Args:
            v:
                Vector to diff from basis
            basis:
                Basis of hyperplane to find diff vector from v with
        Desc:
            Returns vector to v from basis
    '''
    return v - vproj_basis(v, basis)

@arrayargs
def vdist_basis(v, basis):
    '''
        Args:
            v:
                Vector to find distance to basis with
            basis:
                Basis of hyperplane to find distance from v with
        Desc:
            Returns distance from basis to v
    '''
    return norm( vdiff_basis(v, basis) )

@arrayargs
def unit(v):
    '''
        Args:
            v:
                Vector to convert to unit vector
        Desc:
            Returns unit vector form of v if it is of nonzero length,
            else return v of zero length
    '''
    n = norm(v)
    if n != 0:
        return v/n
    else:
        return v

@arrayargs
def remove_colinear(mat):
    '''
        Args:
            mat:
                matrix
        Desc:
            Go along vectors on first axis and remove vectors from end that are colinear
            with a previous vector. Earlier vectors are guaranteed to be kept.
            ex.
                [ [1,1], [2,2] ] -> [ [1,1] ]
    '''

    # Since we will be dealing with large matrices, we convert each vector to its unit vector
    # and hash it. If a vector matches an existing unit vector's direction, then we don't keep it

    unit_vectors = set()
    lin_indep_vectors = []

    for vec in mat:
        u = unit(vec)
        ukey = NdarrayHashKey(u)

        if ukey not in unit_vectors:
            unit_vectors.add(ukey)
            unit_vectors.add(NdarrayHashKey(-u) )
            lin_indep_vectors.append(vec)

    return np.array(lin_indep_vectors)

@arrayargs
def mat_solve_underconstrained(M, b):
    '''
        Args:
            M:
                Matrix consisting of vectors that form a basis. Must be underconstrained.
                First axis holds vectors that form basis, gets transposed internally to match
                numpy's axis ordering.
            b:
                Target vector to determine if reachable from basis.
        Desc:
            Tests whether b is reachable from a linear combination of the vectors of the columns
            of matrix M. First axis of M holds the arrays that make up the basis. M must be
            underconstrained, or this function raises AssertionError.

            Amounts to solving equation M*x = b.

            Since M is underconstrained, we can't invert directly, so we treat x as a vector to
            minimize the length of, and treat M as its Lagrange conditions. Pseudoinverse comes
            out to the form M.T @ inv(M @ M.T). If not solvable, raises np.linalg.LinAlgError.

            Returns x
    '''
    M = M.T # To set basis vectors to vertical so dot products work properly.
    
    assert M.shape[0] < M.shape[1], 'M is not underconstrained, its shape is {}'.format(M.shape)

    pseudoinverse = M.T @ inv(M @ M.T)
    return pseudoinverse @ b
