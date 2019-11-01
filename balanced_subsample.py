def balanced_subsample(x, y, random_state=42):
    # x = dataframe
    # y = series
    # random_state = numeric

    import numpy as np
    import pandas as pd
    import random
    from collections import Counter

    random.seed(random_state)
    np.random.seed(random_state)

    y_unique = np.unique(y)
    y_n = len(y_unique)

    count = np.zeros(y_n, dtype="int")
    for i in range(y_n):
        count[i] = np.sum(y == y_unique[i])

    min_samples = np.min(count)
    balanced_x = pd.DataFrame()
    balanced_y = pd.DataFrame()
    remaining_x = pd.DataFrame()
    remaining_y = pd.DataFrame()

    for i in y_unique:
        tmp_x = x[y.values == i]

        index_sample = np.random.choice(list(tmp_x.index), min_samples,
                                        replace=False)

        rem_index_sample = list(set(tmp_x.index) - set(index_sample))

        balanced_x = balanced_x.append(tmp_x.loc[index_sample, :])
        balanced_y = pd.concat([balanced_y, y.loc[index_sample]])

        remaining_x = remaining_x.append(
            tmp_x.loc[rem_index_sample, :])

        remaining_y = pd.concat(
            [remaining_y, y.loc[rem_index_sample]])

    return balanced_x, balanced_y, remaining_x, remaining_y
