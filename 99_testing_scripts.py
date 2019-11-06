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

print(pp.shape)