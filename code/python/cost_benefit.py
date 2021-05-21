import numpy as np
from tabulate import tabulate
from sklearn import metrics


def latex_printout(model, data, name, table_notes=''):
    train, test = data
    x_train, y_train = train
    x_test, y_test = test
    y_pred = model['estimator'].predict(x_train)
    conf_mat = metrics.confusion_matrix(y_train, y_pred).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    insert_headers(conf_mat)
    
    print('\\begin{table}[!hb]')    
    print('\\centering')
    print(f'\\label{{tab:{name}}}')
    print(f'\\caption{{Confusion matrix for {name} model}} ')
    
    latex_table(conf_mat, conf_pct, f"Train Results with area under the curve score of {model['train_score']:.3f}")
    y_pred = model['estimator'].predict(x_test)
    conf_mat = metrics.confusion_matrix(y_test, y_pred).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    insert_headers(conf_mat)
    latex_table(conf_mat, conf_pct, f"Test Results with area under the curve score of {model['test_score']:.3f}")
    
    print('\\medskip')
    print()
    print('\\footnotesize\\noindent')
    print(
        f"Note: Results from {name} model plotted as a confusion matrix. Each cell contains the number of firms, with row percentages in the paranthesis." + table_notes
    )
    print("\\end{table}")
    
    

def latex_printout_baseline(model, data, name, baseline, table_notes=''):
    train, test = data
    x_train, y_train = train
    x_test, y_test = test
    y_pred = model['estimator'].predict(x_train)
    y_base = baseline['estimator'].predict(x_train)
    conf_mat = metrics.confusion_matrix(y_train, y_pred).tolist()
    base_mat = metrics.confusion_matrix(y_train, y_base).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    base_pct = [[round(i / sum(col), 3) for i in col] for col in base_mat]
    base_mat = list_diff(conf_mat, base_mat)
    base_pct = list_diff(conf_pct, base_pct)
    insert_headers(conf_mat)
    
    print('\\begin{table}[!hb]')    
    print('\\centering')
    print(f'\\label{{tab:{name}}}')
    print(f'\\caption{{Confusion matrix for {name} model}} ')
    latex_table_baseline(conf_mat, conf_pct, base_mat, base_pct, f"Train Results with area under the curve score of {model['train_score']:.3f}")
    print("\\vspace{1cm}")
    
    y_pred = model['estimator'].predict(x_test)
    y_base = baseline['estimator'].predict(x_test)
    conf_mat = metrics.confusion_matrix(y_test, y_pred).tolist()
    base_mat = metrics.confusion_matrix(y_test, y_base).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    base_pct = [[round(i / sum(col), 3) for i in col] for col in base_mat]
    base_mat = list_diff(conf_mat, base_mat)
    base_pct = list_diff(conf_pct, base_pct)
    insert_headers(conf_mat)
    
    latex_table_baseline(conf_mat, conf_pct, base_mat, base_pct, f"Test Results with area under the curve score of {model['test_score']:.3f}")
    print('\\medskip')
    print()
    print('\\footnotesize\\noindent')
    print(
        f"Note: Results from {name} model displayed as a confusion matrix, with results from the Logit model as a baseline. Each cell contains the number of firms, with row percentages in the paranthesis. An additional column is added to compare the type 1 and 2 errors to the Logit baseline. A positive (negative) number means more (fewer) errors." + table_notes
    )
    print("\\end{table}")
    
    
def latex_table(conf_mat, conf_pct, train_test):
    print(f'\\caption*{{{train_test}}}')
    print('\\begin{tabular}{lcc}')
    print('\\hline')
    print(f'{conf_mat[0][0]} & {conf_mat[0][1]} & {conf_mat[0][2]} \\\\')
    print('\\hline')
    print(f'{conf_mat[1][0]} & $\\underset{{({conf_pct[0][0]})}}{{{conf_mat[1][1]}}}$ & $\\underset{{({conf_pct[0][1]})}}{{{conf_mat[1][2]}}}$ \\\\ ')
    print(f'{conf_mat[2][0]} & $\\underset{{({conf_pct[1][0]})}}{{{conf_mat[2][1]}}}$ & $\\underset{{({conf_pct[1][1]})}}{{{conf_mat[2][2]}}}$ \\\\ ')
    print('\\hline')
    print('\\end{tabular}')
    
def latex_table_baseline(conf_mat, conf_pct, base_mat, base_pct, train_test):
    print(f'\\caption*{{{train_test}}}')
    print('\\begin{tabular}{lcc||lc}')
    print('\\hline')
    print(f'{conf_mat[0][0]} & {conf_mat[0][1]} & {conf_mat[0][2]} & & \\makecell{{Baseline \\\\ comparison}} \\\\')
    print('\\hline')
    print(f'{conf_mat[1][0]} & $\\underset{{({conf_pct[0][0]})}}{{{conf_mat[1][1]}}}$ & $\\underset{{({conf_pct[0][1]})}}{{{conf_mat[1][2]}}}$ & \\makecell{{Type 1 \\\\ errors}} & $\\underset{{({base_pct[0][1]})}}{{{base_mat[0][1]}}}$ \\\\ ')
    print(f'{conf_mat[2][0]} & $\\underset{{({conf_pct[1][0]})}}{{{conf_mat[2][1]}}}$ & $\\underset{{({conf_pct[1][1]})}}{{{conf_mat[2][2]}}}$ & \\makecell{{Type 2 \\\\ errors}} & $\\underset{{({base_pct[1][0]})}}{{{base_mat[1][0]}}}$ \\\\ ')
    print('\\hline')
    print('\\end{tabular}')
    
def insert_headers(matrix):
    str_list = ["Non-Bankrupt", "Bankrupt"]
    header_split = f"\\backslashbox{{Actual}}{{Predicted}}"
    for i, col in enumerate(matrix):
        col.insert(0, str_list[i])
    matrix.insert(0, [header_split] + str_list)
    

def list_diff(list1, list2):
    diff = []
    for i in range(len(list1)):
        col = []
        for j in range(len(list1[i])):
            col.append(round(list1[i][j] - list2[i][j], 3))
        diff.append(col)
    return diff


def cost_benefit_analysis(model, data):
    x, y = data
    y_pred = model['estimator'].predict(x)

    # Get the index of bankrupt firms we predict correctly.
    idx_true = (y == y_pred) & (y == 1)

    bankrupt_assets = firm_assets(x, y == 1).sum()
    predicted_assets = firm_assets(x, idx_true).sum()

    save_low = predicted_assets * 0.12
    save_high = predicted_assets * 0.2

    return bankrupt_assets, save_low, save_high, idx_true.sum()


def firm_assets(x, idx):
    try:
        return np.exp(x[idx, 28])
    except TypeError:
        # Ohlsons model
        x = x.to_numpy()
        return np.exp(x[idx, 0])