import os
import pandas as pd
from sklearn import model_selection as selection
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute


def load_data(year: int, path='../../data', **kwargs) -> tuple:
    """Loads x and y for the given year.

    Args:
        year (int): What year to load.
        path (str, optional): Relative or absolute path to folder that has the
        train folder. Defaults to '../../data'.

    Raises:
        ValueError: If you do not give a valid year.

    Returns:
        tuple: Returns the x_train and y_train for given year.
    """
    if year < 1 or year > 5:
        raise ValueError("Year has to be in the range between 1 and 5")
    
    sep = ','
    train_path = os.path.join(path, 'train')    
    test_path = os.path.join(path, 'test')  
    x = '/x_' + str(year) + 'year.txt'
    y = '/y_' + str(year) + 'year.txt'
    x_train = pd.read_csv(train_path + x, sep, **kwargs)
    y_train = pd.read_csv(train_path + y, sep, **kwargs)
    x_test = pd.read_csv(test_path + x, sep, **kwargs)
    y_test = pd.read_csv(test_path + y, sep, **kwargs)
    
    # Squeeze turns vectors from DataFrame to Series.
    return x_train, x_test, y_train.squeeze(), y_test.squeeze()


def split_train_validate(x:pd.DataFrame, y:pd.DataFrame, seed=42) -> tuple:
    test_threshold = 4000
    if x.shape[0] < test_threshold:
        raise ValueError(f'Warning, frame has less than {test_threshold} rows, are you trying to split a test sample, instead of train sample?')
    else:
        return selection.train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=y
        )


def impute_frame(x:pd.DataFrame, random_state=42, **kwargs) -> pd.DataFrame:
    imputer = impute.IterativeImputer(random_state=random_state, **kwargs)
    imputer.fit(x)
    return imputer.transform(x)


