# investigate the trends of label categories

from fpl import fpl
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# region Script Settings
"""
SCRIPT SETTINGS
"""

pandas_display_width = 150
pd.set_option("display.width", pandas_display_width)
pd.set_option("display.max_columns", None)

# endregion

# region Read Data

fixtures = fpl.read_fixtures()
teams = fpl.read_teams()
players = fpl.read_players()
pp = pd.read_csv("data/csv/pp_prepared.csv")

# endregion

# region Scale Numerical Data

num_cols = pp.columns[pp.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
scaler = MinMaxScaler()
pp[num_cols] = scaler.fit_transform(pp[num_cols])

# endregion

Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(pp.values)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
