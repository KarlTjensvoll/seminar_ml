import pandas as pd
import numpy as np

from sklearn import linear_model as linear
from sklearn import model_selection as selection
from sklearn import metrics as metrics

from tabulate import tabulate


def validate_model(
        func, 
        train: tuple,
        test: tuple,
        folds=10, 
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
    val_result = selection.cross_validate(
        func, x_train, y_train, cv=folds, return_estimator=True, **kwargs
    )
    best_fold = np.argmax(val_result.get('test_score'))
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