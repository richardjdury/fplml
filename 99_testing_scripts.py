import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpl import fpl
from balanced_subsample import balanced_subsample
from collections import Counter

# region Script Settings

pandas_display_width = 150
pd.set_option("display.width", pandas_display_width)
pd.set_option("display.max_columns", None)

# endregion

pp = pd.read_csv("data/csv/pp_prepared.csv")
labels = pd.read_csv("data/csv/label_cat.csv")

label_values = np.array(labels.values.reshape(1, -1))

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))


random_state = 42

pp_balanced, labels_balanced, pp_remaining, labels_remaining = \
    balanced_subsample(pp, labels, random_state=random_state)

unique, counts = np.unique(labels_balanced, return_counts=True)
print(dict(zip(unique, counts)))
