import os
import pandas as pd
from sklearn import impute
from sklearn import model_selection
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
        return model_selection.train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=y
        )

def data_pipeline(
        train: pd.DataFrame, test: pd.DataFrame, random_state=42, **kwargs
    ) -> tuple:
    transformer_pipeline = make_pipeline(
        impute.IterativeImputer(random_state=random_state, **kwargs),
        StandardScaler()
    )
    # transformer_pipeline.fit(train)
    train = pd.DataFrame(transformer_pipeline.fit_transform(train))
    test = pd.DataFrame(transformer_pipeline.transform(test))
    return train, test
    

def impute_fit_transform(
        train: pd.DataFrame, test: pd.DataFrame, random_state=42, **kwargs
    ) -> tuple:
    """Fits imputing on the train data, and then fits this both on the train
    and test data.

    Args:
        train (pd.DataFrame): Train data to be fitted and transformed
        test (pd.DataFrame): Test data to be transformed
        random_state (int, optional): Defaults to 42.

    Returns:
        tuple: [description]
    """
    imputer = impute.IterativeImputer(random_state=random_state, **kwargs)
    imputer = imputer.fit(train)
    train = pd.DataFrame(imputer.transform(train))
    test = pd.DataFrame(imputer.transform(test))
    return train, test

def save_files(
        data_list: list, 
        paths: list, 
        file_name: str, 
        prefixes=('x_', 'y_')
    ) -> None:
    """Takes lists of data and paths in order to save them locally.

    Args:
        data_list (list): List of lists, each list containts either train or
        test data.
        paths (list): List of paths for train and test data.
        file_name (str): Name of file, usually a year.
        prefixes (tuple, optional): Defaults to ('x_', 'y_').
    """
    for i, path in enumerate(paths):
        save_paths = [
            os.path.join(path, prefix + file_name) for prefix in prefixes
        ]
        [data.to_csv(save_paths[i], index=False) for i, data in enumerate(data_list[i])]