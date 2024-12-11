import pandas as pd

def load_data_uci(filepath):
    # UCI dataset: lines of "<label>\t<message>"
    df = pd.read_csv(filepath, sep='\t', names=["label", "message"], header=None)
    return df

def load_data_kaggle(filepath):
    # Kaggle spam.csv format often has columns: v1 (label), v2 (message)
    df = pd.read_csv(filepath, encoding='latin-1')
    # Assuming the Kaggle dataset columns: v1 (label) = 'ham'/'spam', v2 (text)
    # Drop extra columns if any (like v3, v4 in older versions of the dataset)
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']  # Rename to match UCI format
    return df

def load_combined_data(uci_path, kaggle_path):
    df_uci = load_data_uci(uci_path)
    df_kaggle = load_data_kaggle(kaggle_path)
    # Combine datasets
    df_combined = pd.concat([df_uci, df_kaggle], ignore_index=True)
    # Remove duplicates if any
    df_combined.drop_duplicates(inplace=True)
    return df_combined
