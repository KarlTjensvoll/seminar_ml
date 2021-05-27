import machine_learning as ml
import data_handler

import numpy as np
from tabulate import tabulate
from sklearn import metrics

import matplotlib
from matplotlib import pyplot as plt


def latex_printout(model, data, name, table_notes=''):
    train, test = data
    x_train, y_train = train
    x_test, y_test = test
    y_pred = model['estimator'].predict(x_train)
    conf_mat = metrics.confusion_matrix(y_train, y_pred).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    insert_headers(conf_mat)
    
    latex_table_start(name)
    latex_table(conf_mat, conf_pct, f"Train Results with area under the curve score of {model['train_score']:.3f}")
    y_pred = model['estimator'].predict(x_test)
    conf_mat = metrics.confusion_matrix(y_test, y_pred).tolist()
    conf_pct = [[round(i / sum(col), 3) for i in col] for col in conf_mat]
    insert_headers(conf_mat)
    latex_table(conf_mat, conf_pct, f"Test Results with area under the curve score of {model['test_score']:.3f}")
    latex_table_end(f"Note: Results from {name} model plotted as a confusion matrix. Each cell contains the number of firms, with row percentages in the paranthesis." + table_notes)
    

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
    
    latex_table_start(name)
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
    latex_table_end(f"Note: Results from {name} model displayed as a confusion matrix, with results from the Logit model as a baseline. Each cell contains the number of firms, with row percentages in the paranthesis. An additional column is added to compare the type 1 and 2 errors to the Logit baseline. A positive (negative) number means more (fewer) errors." + table_notes)


def latex_table_start(caption):
    print('\\begin{table}[!hb]')    
    print('\\centering')
    print(f'\\label{{tab:{caption}}}')
    print(f'\\caption{{Confusion matrix for {caption} model}} ')


def latex_table_end(footnote):
    print('\\medskip')
    print()
    print('\\footnotesize\\noindent')
    print(footnote)
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


def cost_benefit_analysis(model, data, ohlson_data, errorI):
    if model['estimator'].__class__.__name__ == 'LogisticRegression' and model['estimator'].coef_.shape[1] == 8:
        x, y = ohlson_data
    else:
        x, y = data
    y_pred = model['estimator'].predict(x)

    # Get the index of bankrupt firms we predict correctly.
    idx_true = (y == y_pred) & (y == 1)
    bankrupt_assets = firm_assets(x, y == 1).sum()
    predicted_assets = firm_assets(x, idx_true).sum()

    if not errorI:
        save_low = round(predicted_assets * 0.1270 / bankrupt_assets * 100, 2)
        save_high = round(predicted_assets * 0.205 / bankrupt_assets * 100, 2)
        return [str(idx_true.sum()), str(save_low), str(save_high)]
    else:
        idx_type1 = (y_pred == 1) & (y == 0)
        type1_cost = firm_assets(x, idx_type1).sum() * 0.024
        
        save_low = round((predicted_assets * 0.1270 - type1_cost) / bankrupt_assets * 100, 2)
        save_high = round((predicted_assets * 0.205 - type1_cost) / bankrupt_assets * 100, 2)
        
        return [str(idx_true.sum()), str(idx_type1.sum()), str(save_low), str(save_high)]


def firm_assets(x, idx):
    try:
        return np.exp(x[idx, 28])
    except TypeError:
        # Ohlsons model
        x = x.to_numpy()
        return np.exp(x[idx, 0])
    
    
def result_wrapper(model, parameters, train, test, name, baseline=None, scoring='recall'):
    x, y = train[0], train[1]

    # I have to create a fake class in order to make validate model work.
    # This is because it was created with the purpose to work with the
    # grid search algorithm.
    class Object(object):
        pass
    model_obj = Object()
    model_obj.best_estimator_ = model(**parameters).fit(x, y)
    best_model = ml.validate_model(model_obj, train, test, scoring=scoring, supress_print=True)

    # Print latex friendly table
    if not baseline:
        latex_printout(best_model, (train, test), name)
    else:
        latex_printout_baseline(best_model, (train, test), name, baseline)
    return best_model
    

def roc_plotter(
    test_data: tuple, 
    models: list, 
    figsize=(10, 7.5), 
    linewidth=2.5, 
    color_list=['#377eb8', '#4daf4a',
         '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'],
    name_list=['Logit', 'Decision Tree', 'Nerual Network', 'Gradient Booster']
    ):
    x_test, y_test = test_data
    fig, ax = plt.subplots(figsize=figsize)

    # Set options
    ax.set_xlim(-0.005, 1)
    ax.set_ylim(0, 1.008)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'font.size': 16})

    # Allow for multiple models in each plot
    for i, model in enumerate(models):
        y_pred = model['estimator'].predict_proba(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred[:, 1])

        # name = model['estimator'].__class__.__name__
        name = name_list[i]
        ax.plot(fpr, tpr, linewidth=linewidth, label=f"{name}: {model['test_score']:.3f}", color=color_list[i])
        ax.legend(loc='lower right', frameon=False)
    ax.plot([0, 1], [0, 1], linewidth=linewidth, color='grey')  # 45-deg line
    plt.savefig('roc_curves.png')
    plt.show()
    
    
def print_important_coeff(model, n_coeffs):
    important_coeff_arg = np.flip(
        np.argsort(np.abs(model['estimator'].coef_))
    )
    coeff_names = data_handler.ohlson_varnames()

    coeff_list = []
    name_list = []
    for i in range(0, n_coeffs + 1):
        coeff_arg = important_coeff_arg[0, i]
        coeff_list.append(round(model['estimator'].coef_[0, coeff_arg], 3))
        name_list.append(coeff_names[i])
    print_two_rows(coeff_list, name_list)
    return coeff_list, name_list

def print_two_rows(coeff_list, name_list):
    # Split list in two and rezip them
    zipped_list = list(zip(name_list, coeff_list))
    zipped1 = zipped_list[:5]
    zipped2 = zipped_list[5:]
    zipped_list2 = zip(zipped1, zipped2)
    table = tabulate(zipped_list2, tablefmt='latex')

    # Remove tuple and list specifiers
    for char in ["'", "(", ")", "[", "]"]:
        table = table.replace(char, '')

    # Make some small adjustments to get a pretty list
    table = table.replace("ll", "llll")
    table = table.replace(",", " &")
    table = table.replace("cash + short-term securities + receivables - short-term liabilities / operating expenses - depreciation * 365", "defensive interval ratio")
    # table = table.replace("'", '').replace("(", '').replace(")", '').replace("[","").replace("[","")
    print(table)