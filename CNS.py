import numpy
import matplotlib.pyplot as pyplot
from matplotlib import rcParams

import ExactAnalyticalPricing
from FunctionsVectorMatrix import LHS_matrix
from FunctionsVectorMatrix import RHS
rcParams['figure.dpi'] = 100
rcParams['font.size'] = 16
rcParams['font.family'] = 'StixGeneral'

import numpy
import matplotlib.pyplot as pyplot
from matplotlib import rcParams

rcParams['figure.dpi'] = 100
rcParams['font.size'] = 16
rcParams['font.family'] = 'StixGeneral'

T = 1       # expiry time
r = 0.1        # no-risk interest rate
sigma = 0.2    # volatility of underlying asset
E = 100        # exercise price
S_max = 3*E    # upper bound of price of the stock (4*E)

N = 10000       # number of time steps
M = 201        # number of space grids
dt = T/N       # time step
s = numpy.linspace(0, S_max, M)   # spatial grid (stock's price)

# initial condition & boundary condition
C = s - E
C = numpy.clip(C, 0, S_max-E)



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


def CrankNicolson(C, A, N, alpha, beta, gamma, S_max, E):
    """using Crank-Nicolson scheme to solve the Black-Scholes equation for the call option price.

    Arguments:
        C:       array of the price of call option
        A:       LHS coefficient matrix
        N:       total number of time steps
        alpha:   array of coefficients on lower diagnoal
        beta:    array of coefficients on diagnoal
        gamma:   array of coefficients on upper diagnoal
        S_max:   upper bound of stock price
        E:       exercise price

    Returns:
        C:       array of the price of call option
    """
    for t in range(N):
        b = RHS(C, alpha, beta, gamma, S_max, E)
        # use numpy.linalg.solve
        C[1:-1] = solve(A, b)
    return C

from scipy.linalg import solve
N = 200        # number of time steps
dt = T/N       # time step

# initial condition & boundary condition
C = s - E
C = numpy.clip(C, 0, S_max-E)

# calculating the coefficient arrays
index = numpy.arange(1,M-1)

alpha = dt/4 * (r*index - sigma**2*index**2)
beta = dt/2 * (r + sigma**2*index**2)
gamma = -dt/4 * (r*index + sigma**2*index**2)


A = LHS_matrix(M, alpha, beta, gamma)
C_imp = CrankNicolson(C, A, N, alpha, beta, gamma, S_max, E)
print('the price of the call option should be around {}, \
if the price of stock is 20 dollar.'.format(C_imp[116]))

CallPrices4=numpy.array([C_imp[72],C_imp[75],C_imp[80],C_imp[82],C_imp[86],C_imp[90],C_imp[94],C_imp[106],C_imp[116]])
UnderlyingsPrices4=[108,114,120,123,129,135,141,159,174]
Error4 = CallPrices4-ExactAnalyticalPricing.CallPrices1
print(Error4)

pyplot.figure(figsize=(8,5), dpi=100)
pyplot.plot(s[:117], C_imp[:117], color='red', label="BS curve")
pyplot.plot(ExactAnalyticalPricing.s[:117], ExactAnalyticalPricing.C_exact[:117], color='#16537e', ls='--', lw=2, label='Analytical')
pyplot.xlabel('Current Price of the Stock (S)')
pyplot.ylabel('Price of the call option (C)')
pyplot.legend(loc='upper left',prop={'size':15});
pyplot.show()


import pandas as pd
# initialize data of lists.
data = {
        "Asset`s Price": UnderlyingsPrices4,
        "Exact": ExactAnalyticalPricing.CallPrices1,
        "CN": CallPrices4,
        "Error": Error4

        }
# Create DataFrame
df = pd.DataFrame(data)
fig = pyplot.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.table(cellText= df.values, colLabels=df.columns, loc="bottom").scale(1, 4)
ax.plot(UnderlyingsPrices4,Error4)
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.53)
pyplot.show()



