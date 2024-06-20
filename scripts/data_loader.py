

import pandas as pd
import os
import sys

def load_data(path):
    """
    Load the dataset from a CSV or Parquet file.

    Args:
        path (str): Path to the dataset file.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    try:
        # Check if the file is a Parquet file
        if path.endswith('.parquet'):
            df = pd.read_parquet(path, engine='pyarrow')
        # Check if the file is a CSV file
        elif path.endswith('.csv'):
            df = pd.read_csv(path, low_memory=False)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. The dataset file was not found.")
    except pd.errors.ParserError as e:
        print(f"Error: {e}. An error occurred while parsing the dataset.")
    except Exception as e:
        print(f"Error: {e}. An unknown error occurred while loading the dataset.")
    return None


def save_data(df, output_folder, filename):
    """
    Save a pandas DataFrame to a Parquet file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        output_folder (str): The folder path to save the dataset.
        filename (str): The name of the output file.

    Returns:
        str: The full path of the saved file.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        df.to_parquet(output_path)
        print(f"Dataset saved to {output_path}")
        return output_path
    except PermissionError as e:
        print(f"Error: {e}. You do not have permission to write to the output folder.")
    except OSError as e:
        print(f"Error: {e}. An error occurred while creating the output folder.")
    except Exception as e:
        print(f"Error: {e}. An unknown error occurred while saving the dataset.")
    return None
