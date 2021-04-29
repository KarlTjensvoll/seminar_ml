from typing import Callable
from typing import Union
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model as linear
from sklearn import model_selection
from sklearn import metrics
# solving deprecation problems for later versions of sklearn
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
    
from sklearn.exceptions import ConvergenceWarning

from tabulate import tabulate


def grid_cv_report(
        gcv_results: object, x_test:np.array, y_test:np.array
    ) -> None:
    """Summarizes statistics from the cross validated grid search, and passes 
    them through functions to print the summary statistics.

    Args:
        gcv_results (Callable): The estimator (along with all the details from
        each loop), that maximises the givens score.
        table (list): Listlike object that is passed to the tabulate function.
        x_test (np.array): Test features.
        y_test (np.array): Test labels.
    """
    # Gather summary staticits into numpy arrays, and calculate confidence bands.
    means = np.array(gcv_results.cv_results_['mean_test_score'].data, dtype='float')
    std = np.array(gcv_results.cv_results_['std_test_score'].data, dtype='float')
    cf_5 = means - 1.96*std
    cf_95 = means + 1.96*std
    
    # Get the parameter names, and then gather everything into a format
    # that can be passed to the tabulate function.
    params = gcv_results.cv_results_['params']
    if len(params) > 100:
        table = ['Too many combinations to print']
    else:
        table = zip(params, means, cf_5, cf_95)
    
    print_gvc_table(gcv_results, table, x_test, y_test)
    
    
def print_gvc_table(
        gcv_results: Callable, 
        table: Union[list, dict], 
        x_test: np.array, 
        y_test: np.array
    ) -> None:
    """Prints the summary statistics from the cross validated grid search. It
    will print the mean, score upper and lower confidence levels, along with 
    the  set of hyperparameters that gave the specific score. Also print a roc
    curve at the end. 

    Args:
        gcv_results (Callable): The estimator (along with all the details from
        each loop), that maximises the givens score.
        table (list): Listlike object that is passed to the tabulate function.
        x_test (np.array): Test features.
        y_test (np.array): Test labels.
    """
    headers = ['Specification', 'mean score', 'cf - 5%', 'cf - 95%']
    print(f'Machine learning estimator: {gcv_results.estimator}')
    print()
    print(tabulate(table, headers=headers, floatfmt='.3f'))
    print()
    print()
    print(f'Best specification at {gcv_results.best_params_}:')
    print()
    print(metrics.classification_report(
        y_test, gcv_results.predict(x_test), digits=3)
    )
    print()
    plot_gvc_roc(gcv_results, x_test, y_test)
    
    
def plot_gvc_roc(
        gcv_results: Callable, x_test: np.array, y_test: np.array
    ) -> None:
    """Takes the result of the cross validated grid search,
    and plots it in a ROC curve.

    Args:
        gcv_results (Callable): The estimator to pass to the plot_roc_curve function.
        x_test (np.array): Test features.
        y_test (np.array): Test labels.
    """
    metrics.plot_roc_curve(gcv_results, x_test, y_test)
    
    # Also plot a 45-degree line, and adjust the x/y-limit to make it look pretty.
    plt.plot((0, 1), (0, 1), ls='--', c='k')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.0, 1.05])
    plt.show()
    
    
@ignore_warnings(category=ConvergenceWarning)
def search_hyperparameters(
        func: Callable, 
        param_grid: dict, 
        train, 
        scoring: Union[str, dict], 
        refit: str,
        folds=5,
        **kwargs
    ) -> tuple:
    x_train, y_train = train
    grid_search = model_selection.GridSearchCV(
        func, param_grid, cv=folds, scoring=scoring, refit=refit, **kwargs
    )
    grid_results = grid_search.fit(x_train, y_train)
    return grid_results


@ignore_warnings(category=ConvergenceWarning)
def validate_model(
        gcv_results: Callable, 
        train: tuple,
        test: tuple,
        folds=5, 
        shuffle=True, 
        seed=42, 
        **kwargs
    ) -> tuple:
    # Unpack variables
    x_train, y_train = train
    x_test, y_test = test
    
    test_threshold = 4000
    if x_train.shape[0] < test_threshold:
        raise ValueError(f'Warning, frame has less than {test_threshold} rows, are you trying to train on a test sample?')
    
    # Get the best estimator from the gridsearchcv.
    func = gcv_results.best_estimator_

    # Use stratified cross-validation to get the best fitted model.
    # We then find which estimator that performed the best and keep the results from that only.
    val_result = model_selection.cross_validate(
        func, x_train, y_train, cv=folds, 
        return_estimator=True, scoring='recall', return_train_score=True,
        **kwargs
    )
    best_fold = np.argmax(val_result.get('test_recall'))
    results = {item: array[best_fold] for item, array in val_result.items()}

    print_confusion_matrix(results, x_test, y_test)
    return results


def print_confusion_matrix(
        results: object, x_test: np.array, y_test: np.array
    ) -> None:
    # Create confusion matrix
    predicted = results.get('estimator').predict(x_test)
    confusion_mat = metrics.confusion_matrix(y_test, predicted).tolist()

    # Print confusion matrix
    confusion_mat[0].insert(0, "Non-Bankrupt")
    confusion_mat[1].insert(0, "Bankrupt")
    estimator_type = type(results.get('estimator')).__name__
    test_score = results.get('test_score')
    train_score = results.get('train_score')
    headers = ["", "Non-Bankrupt", "Bankrupt"]
    print(f"Confusion matrix from a {estimator_type}, with test-score of {test_score:.3f} and train-score of {train_score:.3f} ")  # Title
    print()
    print(tabulate(confusion_mat, headers=headers))


















def print_confusion_matrix_old(gcv_results, x_test, y_test):
    predicted = gcv_results.best_estimator_.predict(x_test)
    confusion_mat = metrics.confusion_matrix(y_test, predicted).tolist()

    # Print confusion matrix
    confusion_mat[0].insert(0, "Non-Bankrupt")
    confusion_mat[1].insert(0, "Bankrupt")
    estimator_type = gcv_results.scoring
    test_score = gcv_results.best_score_
    headers = ["", "Non-Bankrupt", "Bankrupt"]
    print(f"Confusion matrix from a {estimator_type}, with test-score of {test_score:.3f}")  # Title
    print()
    print(tabulate(confusion_mat, headers=headers))


@ignore_warnings(category=ConvergenceWarning)
def validate_model_old(
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