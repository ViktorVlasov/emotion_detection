import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, 
                 target_col: str, 
                 test_size: float = 0.2, 
                 val_size: float = 0.2, 
                 random_state: int = 42):
        """
        Initialize a DataSplitter instance.

        Parameters:
            target_col (str): The column name containing the target variable.
            test_size (float): The proportion of data to be included in the test set.
            val_size (float): The proportion of data to be included in the validation set.
            random_state (int): The random seed for reproducibility.
        """
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_dataset(self, 
                      df: pd.DataFrame, 
                      balance_classes: bool = True
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            balance_classes (bool): Whether to balance the classes in the dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test DataFrames.
        """
        if balance_classes:
            df = self.balance_classes(df)
        
        num_samples = len(df)
        test_size = int(self.test_size * num_samples)
        val_size = int(self.val_size * num_samples)
        train_size = num_samples - (test_size + val_size)

        train_df, val_df, test_df = np.split(
            df.sample(frac=1, random_state=self.random_state), 
            [train_size, train_size + val_size]
        )

        return pd.DataFrame(train_df), pd.DataFrame(val_df), pd.DataFrame(test_df)


    def balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the classes in the given DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to balance.

        Returns:
            pd.DataFrame: The balanced DataFrame.
        """
        class_counts = df[self.target_col].value_counts()
        min_class_count = class_counts.min()
        balanced_df = pd.concat(
            [df[df[self.target_col] == cls].sample(min_class_count, 
                                                   random_state=self.random_state)
             for cls in class_counts.index]
        )

        return balanced_df


def split_dataset(input_path: str, 
               output_path: str, 
               test_size: float, 
               val_size: float,
               balance_classes: bool):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
        input_path (str): The path to the input dataset.
        output_path (str): The path to save the split datasets.
        test_size (float): The proportion of data to be included in the test set.
        val_size (float): The proportion of data to be included in the validation set.
        balance_classes (bool): Whether to balance the classes in the dataset.
    """

    df = pd.read_csv(input_path)

    splitter = DatasetSplitter(target_col='labels', 
                               test_size=test_size, 
                               val_size=val_size)

    train_df, val_df, test_df = splitter.split_dataset(df, balance_classes)


    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    train_df.to_csv(f'{output_path}/train.csv', index=False)
    val_df.to_csv(f'{output_path}/val.csv', index=False)
    test_df.to_csv(f'{output_path}/test.csv', index=False)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Dataset Splitter')
    parser.add_argument('--input', type=str, help='Path to the input dataset')
    parser.add_argument('--output', type=str, help='Path to save the split datasets')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data in the test set')
    parser.add_argument('--val-size', type=float, default=0.2, help='Proportion of data in the validation set')
    parser.add_argument('--balance-classes', action='store_true', help='Whether to balance the classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    split_dataset(args.input, args.output, args.test_size, args.val_size, args.balance_classes)