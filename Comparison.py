import numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.pyplot as pyplot

import ExactAnalyticalPricing
from ExactAnalyticalPricing import UnderlyingsPrices1 as UnderlyingsPrices
from Implicit import Error2
from ForwardEES import Error3
from CNS import Error4

import pandas as pd
# initialize data of lists.
data = {
        "Asset`s Price": UnderlyingsPrices,
        "Implicit-Error": Error2,
        "Explicit-Error": Error3,
        "CN-Error": Error4

        }
# Create DataFrame
df = pd.DataFrame(data)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.table(cellText= df.values, colLabels=df.columns, loc="bottom").scale(1, 4)
ax.plot(UnderlyingsPrices,Error2, color = "r", label = "Implicit-Error")
ax.plot(UnderlyingsPrices,Error3, color = "b", label = "Explicit-Error", ls='--')
ax.plot(UnderlyingsPrices,Error4, color = "g", label = "CN-Error")
ax.legend(loc='upper right',prop={'size':8});
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.53)
pyplot.show()
