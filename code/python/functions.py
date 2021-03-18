from sklearn import model_selection
import numpy as np
import pandas as pd
import os

from sklearn.utils import shuffle

# Loop over all raw files.
# Use only those that end with .txt, as these are clean.
# Load them into a dataframe, and then split into test and train set.

# Seed is taken from random.org
seed = 418409376

def load_data(year: int, share: float, shuffle: bool, seed: int):
    """
    Loads the dataset and splits it as preferred. 
    Use this if you want to tweak size of test/train or shuffle     

    Args:
        year (str): which year of forecasting period we want to load
        share (float): share of data to be in test dataset
        shuffle (bool): shuffle or not
        seed (int): Seed to use in random number generator for model_selection.
        
    Output:
        1) x_train
        2) y_train
        3) x_test
        4) y_test
    """
    
    raw_path = 'data/raw_do_not_touch/'
    file_name = '{}year.txt'.format(year)   # only dealing with the cleaned .txt files
    
    assert os.path.isfile(os.path.join(raw_path, file_name))
    path = os.path.join(raw_path, file_name)
    
    df = pd.read_csv(path, delimiter=',')
            
    # All files should have at 65 columns.
    assert df.shape[1] == 65
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # We stratify on y, to make sure that we have proportionally
    # the same amount of bankrupt firm in train and test.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, 
        y, 
        test_size=share, 
        random_state=seed, 
        shuffle=shuffle, 
        stratify=y)
        # Make sure that the number of bankrupt firms are about the same in
        # train and test.
    assert np.isclose(y_train.mean(), y_test.mean(), atol=0.0005)
        
    return x_train, y_train, x_test, y_test
            





