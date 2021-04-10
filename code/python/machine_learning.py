from typing import Callable
import pandas as pd
import numpy as np

from sklearn import linear_model as linear
from sklearn import model_selection
from sklearn import metrics 
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from tabulate import tabulate


@ignore_warnings(category=ConvergenceWarning)
def search_hyperparameters(
        func: Callable, param_grid: dict, train, folds=5
    ) -> tuple:
    x_train, y_train = train
    grid_search = model_selection.GridSearchCV(
        func, param_grid, cv=folds)
    grid_results = grid_search.fit(x_train, y_train)
    return grid_results


@ignore_warnings(category=ConvergenceWarning)
def validate_model(
        func: Callable, 
        train: tuple,
        test: tuple,
        folds=5, 
        shuffle=True, 
        seed=42, 
        **kwargs
    ) -> tuple:
    test_threshold = 4000
    if train[0].shape[0] < test_threshold:
        raise ValueError(f'Warning, frame has less than {test_threshold} rows, are you trying to train on a test sample?')

    # Unpack variables
    x_train, y_train = train
    x_test, y_test = test

    # Use stratified cross-validation to get the best fitted model.
    # We then find which estimator that performed the best and keep the results from that only.
    val_result = model_selection.cross_validate(
        func, x_train, y_train, cv=folds, 
        return_estimator=True, scoring='recall', **kwargs
    )
    best_fold = np.argmax(val_result.get('test_recall'))
    results = {item: array[best_fold] for item, array in val_result.items()}

    # Create confusion matrix
    predicted = results.get('estimator').predict(x_test)
    confusion_mat = metrics.confusion_matrix(y_test, predicted).tolist()

    # Print confusion matrix
    confusion_mat[0].insert(0, "Non-Bankrupt")
    confusion_mat[1].insert(0, "Bankrupt")
    estimator_type = type(results.get('estimator')).__name__
    test_score = results.get('test_score')
    headers = ["", "Non-Bankrupt", "Bankrupt"]
    print(f"Confusion matrix from a {estimator_type}, with test-score of {test_score:.3f}")  # Title
    print()
    print(tabulate(confusion_mat, headers=headers))
    return results