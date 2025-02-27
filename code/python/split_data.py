from sklearn import model_selection
import numpy as np
import pandas as pd
import os
import data_handler

# from sklearn.utils import shuffle

# Loop over all raw files.
# Use only those that end with .txt, as these are clean.
# Load them into a dataframe, and then split into test and train set.
def split_data(
        raw_path: str, train_path: str, test_path: str, 
        seed: int, max_iter=100, add_indicator=False
    ) -> None:
    """Loop over all the files in raw_path that ends with .txt. Then split it
    into train and test data sets, with 20% of data split into train. Then
    save them in train and test folders respectively, as .txt files. They will
    have name x_1, x_2, ... or y_1, y_2, ... etc. where the number corresponds
    to the year.

    Args:
        raw_path (str): Path that has the raw data files.
        train_path (str): Path to store the train data files.
        test_path (str): Path to store the test data files.
        seed (int): Seed to use in random number generator for model_selection.
    """
    for file in os.listdir(raw_path):
        if file.endswith('.txt'):
            path = os.path.join(raw_path, file)
            df = pd.read_csv(path, delimiter=',')
            
            # All files should have at 65 columns.
            assert df.shape[1] == 65
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            x = x.to_numpy()

            # We stratify on y, to make sure that we have proportionally
            # the same amount of bankrupt firm in train and test.
            x_train, x_test, y_train, y_test = model_selection.train_test_split(
                x, 
                y, 
                test_size=0.2, 
                random_state=seed, 
                shuffle=True, 
                stratify=y
            )
            # Make sure that the number of bankrupt firms are about the same in
            # train and test.
            assert np.isclose(y_train.mean(), y_test.mean(), atol=0.0005)
            
            # Impute missing. We do this seperately, so that no information
            # from train data spills over to test data.
            print(f"Imputing train for {file}.")
            x_train, x_test = data_handler.data_pipeline(
                x_train, x_test, max_iter=max_iter, add_indicator=add_indicator
            )
            x_train = pd.DataFrame(x_train)
            x_test = pd.DataFrame(x_test)
            
            data_to_save = [[x_train, y_train], [x_test, y_test]]
            paths = [train_path, test_path]
            data_handler.save_files(data_to_save, paths, file)
    


# Seed is taken from random.org
seed = 418409376
raw_path = '../../data/raw_do_not_touch'
train_path = '../../data/train'
test_path = '../../data/test'

split_data(raw_path, train_path, test_path, seed)