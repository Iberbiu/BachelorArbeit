
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

N = 10000      # number of time steps
M = 201        # number of space grids
dt = T/N       # time step
s = numpy.linspace(0, S_max, M)   # spatial grid (stock's price)

from scipy.stats import norm

C_exact = numpy.zeros(M)

d1 = (numpy.log1p(s/E) + (r+0.5*sigma**2)*T) / (sigma * numpy.sqrt(T))
d2 = d1 - (sigma * numpy.sqrt(T))
C_exact = s * norm.cdf(d1) - E*numpy.exp(-r*T) * norm.cdf(d2)
C_exact = numpy.clip(C_exact, 0, numpy.inf)



pyplot.figure(figsize=(8,5), dpi=100)
pyplot.plot(s, C_exact, color='#16537e', ls='-', lw=2, label='Analytical')
pyplot.xlabel('Current Price of the Stock (S)')
pyplot.ylabel('Price of the call option (C)')
pyplot.legend(loc='upper left',prop={'size':15});
pyplot.show()

print ('Exact Solution: C(S=20, t=0) = {}'.format(C_exact[134]))
CallPrices1=numpy.array([C_exact[72],C_exact[75],C_exact[80],C_exact[82],C_exact[86],C_exact[90],C_exact[94],C_exact[106],C_exact[116]])
UnderlyingsPrices1=[108,114,120,123,129,135,141,159,174]
print(CallPrices1)
