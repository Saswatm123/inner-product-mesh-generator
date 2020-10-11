from utils.general import to_array

class Shape:
    def __init__(self):
        pass

class Sphere(Shape):
    '''
        Interface for sphere for convenience, taking in
        Basis of subspace containing sphere, center, and radius.
    '''
    def __init__(self, basis, center, radius):
        '''
            Args:
                basis:
                    Basis of plane containing Sphere
                center:
                    Center of Sphere
                radius:
                    Radius of Sphere
        '''
        super().__init__()
        self.basis = to_array(basis)
        self.center = to_array(center)
        self.radius = radius

    def __str__(self):
        return 'Sphere object\n\tCenter = {}\n\tRadius = {}\n\tBasis = \n\t{}'.format(
            self.center,
            self.radius,
            str(self.basis).replace('\n', '\n\t')
        )

class HyperPlane(Shape):
    '''
        Interface for hyperplane for convenience, taking in
        Basis of hyperplane and Point on hyperplane for shifting.
    '''
    def __init__(self, basis, point):
        '''
            Args:
                basis:
                    Basis of hyperplane
                point:
                    Any point on the hyperplane, for shifting purposes
        '''
        super().__init__()
        self.basis = to_array(basis)
        self.point = to_array(point)

    def __str__(self):
        return 'HyperPlane object\n\tPoint = {}\n\tBasis = \n\t{}'.format(
            self.point,
            str(self.basis).replace('\n', '\n\t')
        )
