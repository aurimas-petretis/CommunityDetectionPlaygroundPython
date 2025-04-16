import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog



def problem_5():
    print('Problem 5. Linear optimization')
    print('maximize | f(x) | x')

    c = [2, -3, 0, -5]
    cmax = [-2, 3, 0, 5]
    A = [[-1, 1, -1, -1], [2, 4, 0, 0], [0, 0, 1, 1]]
    At = numpy.transpose(A)
    b1 = [8, 10, 3]
    b2 = [6, 3, 7]

    result1 = problem_5_go(c, A, b1, maximize=False)
    result1d = problem_5_go(b1, -At, c, maximize=False)
    result2 = problem_5_go(c, A, b2, maximize=False)
    result2d = problem_5_go(b2, -At, c, maximize=False)

    assert np.allclose((np.dot(A, result1.x) - b1) * result1d.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result1d.x) - c) * result1.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax
    assert np.allclose((np.dot(A, result2.x) - b2) * result2d.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result2d.x) - c) * result2.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax

    result1max = problem_5_go(cmax, A, b1, maximize=True)
    result1dmax = problem_5_go(b1, -At, cmax, maximize=True)
    result2max = problem_5_go(cmax, A, b2, maximize=True)
    result2dmax = problem_5_go(b2, -At, cmax, maximize=True)

    assert np.allclose((np.dot(A, result1max.x) - b1) * result1dmax.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result1dmax.x) - cmax) * result1max.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax
    assert np.allclose((np.dot(A, result2max.x) - b2) * result2dmax.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result2dmax.x) - cmax) * result2max.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax


def problem_5_go(c, A, b, maximize=False):
    # bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)] # not needed, this is set by default
    result = linprog(c, A_ub=A, b_ub=b)
    print(maximize, round(-result.fun if maximize else result.fun, 4), result.x)
    return result


def main():
    print('Starting the program...')
    problem_5()
    print('Program finished')


main()
