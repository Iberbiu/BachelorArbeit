import numpy as np
from scipy import sparse
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
import scipy as scp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm, pyplot

import ExactAnalyticalPricing
r = 0.1;
sig = 0.2
S0 = 100;
X0 = np.log(S0)
K = 100;
Texpir = 1

Nspace = 201  # M space steps
Ntime = 10000  # N time steps

S_max = 3 * float(K)
S_min = float(K) / 3

x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization

T, dt = np.linspace(0, Texpir, Ntime, retstep=True)  # time discretization
Payoff = np.maximum(np.exp(x) - K, 0)  # Call payoff

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms

V[:, -1] = Payoff  # terminal conditions
V[-1, :] = np.exp(x_max) - K * np.exp(-r * T[::-1])  # boundary condition
V[0, :] = 0







# construction of the tri-diagonal matrix D
sig2 = sig * sig;
dxx = dx * dx

a = ((dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx))
b = (1 + dt * (sig2 / dxx + r))
c = (-(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx))

D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

# Backward iteration
for i in range(Ntime - 2, -1, -1):
    offset[0] = a * V[0, i]
    offset[-1] = c * V[-1, i];
    V[1:-1, i] = spsolve(D, (V[1:-1, i + 1] - offset))

# finds the option at S0
oPrice = np.interp(np.log(200), x, V[:, 0])
print('the price of the call option should be around {}'.format(oPrice))
CallPrices2 = np.array([np.interp(np.log(108), x, V[:, 0]),np.interp(np.log(114), x, V[:, 0]),np.interp(np.log(120), x, V[:, 0]),np.interp(np.log(123), x, V[:, 0]),np.interp(np.log(129), x, V[:, 0]),np.interp(np.log(135), x, V[:, 0]),np.interp(np.log(141), x, V[:, 0]),np.interp(np.log(159), x, V[:, 0]),np.interp(np.log(174), x, V[:, 0])])
UnderlyingsPrices2=[108,114,120,123,129,135,141,159,174]
Error2 = CallPrices2-ExactAnalyticalPricing.CallPrices1
print(Error2)
print(CallPrices2)
print(UnderlyingsPrices2)

S = np.exp(x)
"""
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)

ax1.plot(S, Payoff, color='blue', label="Payoff")
ax1.plot(S, V[:, 0], color='red', label="BS curve Implicit")
ax1.plot(ExactAnalyticalPricing.s, ExactAnalyticalPricing.C_exact, color='#FFFF00', ls='--', lw=2, label='Analytical')
ax1.set_xlim(0, 300);
ax1.set_ylim(0, 250)
ax1.set_xlabel("S");
ax1.set_ylabel("price")
ax1.legend(loc='upper left');
ax1.set_title("BS price at t=0")
ax2 = fig.add_subplot(122, projection='3d')
X, Y = np.meshgrid(T, S)
ax2.plot_surface(Y, X, V, cmap=cm.ocean)
ax2.set_title("BS price surface")
ax2.set_xlabel("S");
ax2.set_ylabel("t");
ax2.set_zlabel("V")
ax2.view_init(30, -100)  # this function rotates the 3d plot

plt.show()
"""
pyplot.figure(figsize=(8,5), dpi=100)
pyplot.plot(S, Payoff, color='#e98a1f', label="Payoff")
pyplot.plot(S, V[:, 0], color='red', label="Implicit Approximation")
pyplot.plot(ExactAnalyticalPricing.s, ExactAnalyticalPricing.C_exact, color='#16537e', ls='--', lw=2, label='Analytical')
pyplot.xlabel('Current Price of the Stock (S)')
pyplot.ylabel('Price of the call option (C)')
pyplot.legend(loc='upper left',prop={'size':15});
pyplot.show()


import pandas as pd
# initialize data of lists.
data = {
        "Asset`s Price": UnderlyingsPrices2,
        "Exact": ExactAnalyticalPricing.CallPrices1,
        "Implicit": CallPrices2,
        "Error": Error2

        }
# Create DataFrame
df = pd.DataFrame(data)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.table(cellText= df.values, colLabels=df.columns, loc="bottom").scale(1, 4)
ax.plot(UnderlyingsPrices2,Error2)
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.53)
pyplot.show()
