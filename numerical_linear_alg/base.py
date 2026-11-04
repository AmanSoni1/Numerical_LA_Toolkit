"""
Base class shared by all linear-system solvers.
"""


class LinearSolver:

    # Name of the solver
    name = "Linear solver"

    def solve(self, A, b):
        """
        Solve the linear system Ax = b.

        Every child class should implement this method.
        """
        raise NotImplementedError("Solve method must be implemented")

    def __call__(self, A, b):
        """
        Allow the solver object to be called like a function.
        """
        return self.solve(A, b)

    def __repr__(self):
        """
        Return the name of the solver object.
        """
        return self.__class__.__name__ + "()"