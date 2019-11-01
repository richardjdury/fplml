# random forest alogrithm to predict score

from fpl import fpl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.naive_bayes import GaussianNB
from balanced_subsample import balanced_subsample
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

random_state = 42
np.random.seed(random_state)
random.seed(random_state)
from sklearn.feature_selection import SelectFromModel


# region Script Settings
"""
SCRIPT SETTINGS
"""

pandas_display_width = 150
pd.set_option("display.width", pandas_display_width)
pd.set_option("display.max_columns", None)

# endregion

# region Read Data

pp = pd.read_csv("data/csv/pp_prepared.csv")
labels_value = pd.read_csv("data/csv/label_values.csv")
labels_cat = pd.read_csv("data/csv/label_cat.csv")

# endregion

# region Choose Label Type

label_type = "cat"

if label_type == "cat":
    labels = labels_cat.copy()
    label_names = ["Poor", "Average", "Good", "Excellent"]
elif label_type == "reg":
    labels = labels_value.copy()

# endregion

# region Scale Numerical Data

num_cols = pp.columns[pp.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
scaler = MinMaxScaler()
pp[num_cols] = scaler.fit_transform(pp[num_cols])

# endregion

# region Remove Correlated Features

corr_matrix = pp.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

pp = pp.drop(pp[to_drop], axis=1)

# endregion

# region Balanced Subsampling

pp_balanced, labels_balanced, pp_remaining, labels_remaining = \
    balanced_subsample(pp, labels, random_state=random_state)

# endregion

# region Test Train Split

if label_type == "cat":

    split = StratifiedShuffleSplit(n_splits=10, test_size=0.25,
                                   random_state=random_state)

    for train_index, test_index in split.split(pp_balanced.values,
                                               labels_balanced.values):
        X_train, X_test = pp_balanced.values[train_index], \
                          pp_balanced.values[test_index]
        Y_train, Y_test = labels_balanced.values[train_index], \
                          labels_balanced.values[test_index]

    baseline_error = np.sum(Y_train == mode(Y_train)) / len(X_train)

    Y_train_mode = mode(Y_train)
    baseline_predictions = np.ones((len(Y_train), 1)) * Y_train_mode[0][0]

    print("BASELINE CLASSIFICATION REPORT")
    print(classification_report(Y_train, baseline_predictions,
                                target_names=label_names))


elif label_type == "reg":

    print("regression")

# endregion

# region Run Random Forest

if label_type == "cat":

    # region Define Classifier

    classifier = RandomForestClassifier(n_estimators=1000,
                                        random_state=random_state,
                                        max_depth=5,
                                        max_features=10,
                                        min_samples_leaf=5)

    # endregion

    # region Train Model and plot importance

    classifier.fit(X_train, Y_train.ravel())

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 2)
    importance = classifier.feature_importances_
    importance = pd.DataFrame(importance, index=pp.columns,
                              columns=["Importance"])

    importance["Std"] = np.std([tree.feature_importances_
                                for tree in classifier.estimators_], axis=0)

    x = range(importance.shape[0])
    y = importance.iloc[:, 0]
    yerr = importance.iloc[:, 1]
    plt.barh(pp.columns, y, xerr=yerr, align="center")
    plt.xlabel("Importance")

    # endregion

    # region Fit Model

    train_predictions = classifier.predict(X_train)

    training_error = np.sum(train_predictions == Y_train.ravel()) / len(
        X_train)

    print(classification_report(train_predictions, Y_train))

    # endregion

    # region Test Model

    test_predictions = classifier.predict(X_test)

    test_error = np.sum(test_predictions == Y_test.ravel()) / len(X_test)

    print(classification_report(test_predictions, Y_test))

    # endregion

    # region Confusion Matrix

    train_cm = confusion_matrix(Y_train, train_predictions)
    train_cm_df = pd.DataFrame(train_cm, index=label_names,
                               columns=label_names)

    plt.subplot(2, 2, 3)
    sn.heatmap(train_cm_df, annot=True, fmt="d", robust=True)
    plt.ylim(-0.5, 4.5)
    plt.xlim(-0.5, 4.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Training")

    plt.subplot(2, 2, 4)
    test_cm = confusion_matrix(Y_test, test_predictions)
    test_cm_df = pd.DataFrame(test_cm, index=label_names,
                              columns=label_names)
    sn.heatmap(test_cm_df, annot=True, fmt="d", robust=True)
    plt.ylim(-0.5, 4.5)
    plt.xlim(-0.5, 4.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Test")

    plt.show()

    # endregion

elif label_type == "reg":

    print("regression")

# endregion