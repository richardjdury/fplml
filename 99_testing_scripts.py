import pandas as pd
import numpy as np
from fpl import fpl
from balanced_subsample import balanced_subsample

# region Script Settings

pandas_display_width = 150
pd.set_option("display.width", pandas_display_width)
pd.set_option("display.max_columns", None)

# endregion

pp = pd.read_csv("data/csv/pp_prepared.csv")
labels = pd.read_csv("data/csv/label_cat.csv")

pp_balanced, labels_balanced, pp_remaining, labels_remaining = \
    balanced_subsample(pp, labels)