import numpy as np
import matplotlib.pyplot as pyplot
import ExactAnalyticalPricing

NT = 10000
NX = 201
L = 300
K = 100
sigmap = 0.2

r = 0.1
u = [None] * NX
def phiT(s):
    if(s > K):
        return s -K
    else:
        return 0
def PDEfiniteDiff():
    dx = L/(NX-1)
    dt = 1./NT

    t = 0
    for i in range(0,NX):
        u[i] = phiT(i * dx)
    for j in range(0,NT):
        t += dt
        for i in range(0, NX-1):
            x = i * dx
            u[i] += (0.5 * sigmap * x * sigmap * x * (u[i + 1] - 2 * u[i] + u[i-1])/ dx / dx + r * x * (u[i+1] - u[i])/dx - r * u[i]) * dt
        u[NX -1] = L - K *  np.exp(- r * t)
        u[0] = 0
    return u
data = PDEfiniteDiff()
S_max = 300    # upper bound of price of the stock (4*E)
M=200
s = np.linspace(0, S_max, M+1)   # spatial grid (stock's price)
#Ndrysho emrat e boshteve
pyplot.figure(figsize=(8,5), dpi=100)
pyplot.plot(s,data,color='#e98a1f', ls='--', lw=3, label='Explicit');
pyplot.plot(ExactAnalyticalPricing.s, ExactAnalyticalPricing.C_exact, color='#16537e', ls='--', lw=2, label='Analytical')
pyplot.xlabel('Current Price of the Stock (S)')
pyplot.ylabel('Price of the call option (C)')
pyplot.legend(loc='upper left',prop={'size':15});
pyplot.show()
print(data)

CallPrices3 = np.array([data[72],data[75],data[80],data[82],data[86],data[90],data[94],data[106],data[116]])
UnderlyingsPrices3=[108,114,120,123,129,135,141,159,174]
Error3 = CallPrices3-ExactAnalyticalPricing.CallPrices1
print(Error3)

"""
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(121)
ax1= fig.add_plot

ax1.plot(S, Payoff, color='blue', label="Payoff")
ax1.plot(S, V[:, 0], color='red', label="BS curve")
ax1.plot(ExactAnalyticalPricing.s, ExactAnalyticalPricing.C_exact, color='#FFFF00', ls='--', lw=2, label='Analytical')
ax1.set_xlim(0, 200);
ax1.set_ylim(0, 120)
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




import pandas as pd
# initialize data of lists.
data = {
        "Asset`s Price": UnderlyingsPrices3,
        "Exact": ExactAnalyticalPricing.CallPrices1,
        "Explicit": CallPrices3,
        "Error": Error3

        }
# Create DataFrame
df = pd.DataFrame(data)
fig = pyplot.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.table(cellText= df.values, colLabels=df.columns, loc="bottom").scale(1, 4)
ax.plot(UnderlyingsPrices3,Error3)
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.53)
pyplot.show()
