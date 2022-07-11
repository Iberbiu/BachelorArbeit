import numpy

def LHS_matrix(M, alpha, beta, gamma):
    """generate and return the LHS coefficient matrix A.

    Arguments:
        M:       total number of spatials grids
        alpha:   array of coefficients on lower diagnoal
        beta:    array of coefficients on diagnoal
        gamma:   array of coefficients on upper diagnoal

    Returns:
        A:       LHS coefficient matrix
    """
    # diagonal
    d = numpy.diag(1 + beta)
    # upper diagonal
    ud = numpy.diag(gamma[:-1], 1)
    # lower diagonal
    ld = numpy.diag(alpha[1:], -1)

    A = d + ud + ld
    return A
def RHS(C, alpha, beta, gamma, S_max, E):
    """generate and return the RHS vector b.

    Arguments:
        C:       array of the price of call option at previous time step
        alpha:   array of coefficients on lower diagnoal
        beta:    array of coefficients on diagnoal
        gamma:   array of coefficients on upper diagnoal
        S_max:   upper bound of stock price
        E:       exercise price

    Returns:
        b:       RHS vector
    """
    # diagonal of A_star
    d = numpy.diag(1 - beta)
    # upper diagonal of A_star
    ud = numpy.diag(-gamma[:-1], 1)
    # lower diagonal of A_star
    ld = numpy.diag(-alpha[1:], -1)

    A_star = d + ud + ld
    b = numpy.dot(A_star, C[1:-1])
    # add BC for the right bound (the last element)
    b[-1] += -2 * gamma[-1] * (S_max - E)

    return b