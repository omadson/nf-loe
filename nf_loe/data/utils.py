import pickle
from pathlib import Path
import pandas as pd

def load_pickle_data(dataset_path, num_variables: int=1):
    with open(Path(dataset_path), "rb") as f:
        if num_variables != 1:
            return pd.DataFrame(
                pickle.load(f).reshape((-1, num_variables))
            )
        return pd.DataFrame(
            pickle.load(f).reshape((-1))
        )