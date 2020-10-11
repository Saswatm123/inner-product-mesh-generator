from math import sqrt
from numpy.linalg import norm
from utils.vector_math import vdiff_basis
from utils.shapes import Sphere, HyperPlane

def sphere_spanned_hyperplane_intersection(S, H):
    '''
        Args:
            S:
                Sphere
            H:
                HyperPlane whose basis and centering point are set up so that
                all points on H are reachable from S's basis considering its centering
                point
        Desc:
            Finds intersection sphere of S and H, where H is hyperplane
            reachable from S's basis taking into consideration both their centerings.
            Assumes H's subspace <= S's subspace.
    '''

    # First, we find shortest vector from HyperPlane to Sphere center.

    Hpt_to_c = S.center - H.point

    H_to_c = vdiff_basis(Hpt_to_c, H.basis)

    # If distance to HyperPlane > Sphere's radius, then no intersection.

    if norm(H_to_c) > S.radius:
        return False

    # Else, there is some intersection of either 0 dimensions in the case of H being tangential to S,
    # rank(Sphere.basis) - 1 in the case of H going through the hull of S, or
    # rank(Sphere.basis) in the case that H and S share the same subspace

    return Sphere(
        basis = H,
        center = S.center + (-H_to_c),
        radius = sqrt(S.radius**2 - norm(H_to_c)**2 )
    )
