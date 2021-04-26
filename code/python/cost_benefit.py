import numpy as np
from tabulate import tabulate
from sklearn import metrics


def latex_printout(model, data):
    x, y = data
    y_pred = model['estimator'].predict(x)
    confusion_mat = metrics.confusion_matrix(y, y_pred).tolist()
    confusion_mat[0].insert(0, "Non-Bankrupt")
    confusion_mat[1].insert(0, "Bankrupt")
    headers = ["", "Non-Bankrupt", "Bankrupt"]
    print(tabulate(confusion_mat, headers=headers, tablefmt='latex'))


def cost_benefit_analysis(model, data):
    x, y = data
    y_pred = model['estimator'].predict(x)

    # Get the index of bankrupt firms we predict correctly.
    idx_true = (y == y_pred) & (y == 1)

    bankrupt_assets = firm_assets(x, y == 1).sum()
    predicted_assets = firm_assets(x, idx_true).sum()

    save_low = predicted_assets * 0.12
    save_high = predicted_assets * 0.2

    return bankrupt_assets, save_low, save_high


def firm_assets(x, idx):
    try:
        return np.exp(x[idx, 28])
    except TypeError:
        # Ohlsons model
        x = x.to_numpy()
        return np.exp(x[idx, 0])