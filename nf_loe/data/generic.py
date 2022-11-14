from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import make_moons
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


data_path = (
    Path(__file__)
    .absolute()
    .parents[2] / 'data'
)

def create_window_df(df, window_size):
    if window_size not in [None, 1]:
        num_variables = df.shape[1]
        windows = []
        index = []
        for window in df.rolling(window=window_size):
            values = window.to_numpy().flatten()
            if len(values) == num_variables * window_size:
                index.append(window.index[0])
                windows.append(values)
        return pd.DataFrame(windows, index=index)
    return df

class SyntheticDataset(Dataset):
    """Generate a Synthetic dataset."""

    def __init__(self, num_rows, contamination=.01, random_state=42):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        n_outliers = int(contamination * num_rows)
        n_inliers = num_rows - n_outliers
        X =  make_moons(n_samples=n_inliers, noise=0.05)[0] - [.5, .25]
        rng = np.random.RandomState(random_state)
        outliers = rng.uniform(low=-3, high=3, size=(n_outliers, 2))
        y = np.concatenate([
            np.ones(X.shape[0]),
            np.zeros(n_outliers)
        ],axis=0)
        X = np.concatenate([X, outliers], axis=0)
        X, y = shuffle(X, y, random_state=random_state)
        self.samples = torch.from_numpy(X.astype(np.float32))
        self.labels = torch.from_numpy(y[np.newaxis].T.astype(np.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        print('indices', idx)
        data = {
            'sample': self.samples[idx, :],
            'label': self.labels[idx, :]
        }
        return data

class GenericDataset(Dataset):
    """Generic dataset"""

    def __init__(self, csv_path=None, data_frame=None, target_column=None, index_col=None, window_size=None, scaler=None):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.window_size = window_size
        self.target_column = target_column
        if data_frame is not None:
            self.df = data_frame
        else:
            self.df = pd.read_csv(csv_path, index_col=index_col)
        self.df = ( 
            self
            .df
            .pipe(lambda df: df.fillna(df.interpolate(method='polynomial', order=2)))
            .pipe(create_window_df, self.window_size)
        )
        self.scaler = StandardScaler()
        if scaler != None:
            self.scaler = scaler
        # Save target and predictors
        if self.target_column:
            data = self.df.drop(self.target_column, axis=1).values.astype(np.float32)
            if scaler != None:
                data = self.scaler.transform(data)
            else:
                data = self.scaler.fit_transform(data)
                
            self.X = torch.from_numpy(data)
            self.y = torch.from_numpy(self.df[[self.target_column]].values.astype(np.float32))
        else:
            data = self.df.values.astype(np.float32)
            if scaler != None:
                data = self.scaler.transform(data)
            self.X = torch.from_numpy(data)
        # self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0)+1e-5)
        # self.X[:, 0] = 1e-10
            

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if self.target_column:
            data = {
                'sample': self.X[idx, :],
                'label': self.y[idx, :]
            }
        else:
            data = {
                'sample': self.X[idx, :]
            }
        return data
    
    @property
    def shape(self):
        return self.X.shape
        

class WindowDataset(Dataset):
    """Generic dataset"""

    def __init__(self, csv_path=None, data_frame=None, target_column=None, index_col=None, serie_col=None):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.serie_col = serie_col
        self.target_column = target_column
        if data_frame is not None:
            self.df = data_frame
        else:
            self.df = pd.read_csv(csv_path, index_col=index_col)
        self.df = ( 
            self
            .df
            .pipe(lambda df: df.fillna(df.interpolate(method='polynomial', order=2)))
        )
        self.scaler = StandardScaler()
        # Save target and predictors
        data = self.df.drop(columns=[self.target_column, self.serie_col], axis=1).values.astype(np.float32)
        self.X = torch.from_numpy(data)
        self.y = torch.from_numpy(self.df[[self.target_column]].values.astype(np.float32))
        self.id = torch.from_numpy(self.df.index.to_numpy().astype(np.int))
        self.serie = torch.from_numpy(self.df[[self.serie_col]].values.astype(np.int))
            

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return {
            'sample': self.X[idx, :],
            'label': self.y[idx, :],
            'serie': self.serie[idx, :],
            'point': self.id[idx],
        }
    
    @property
    def shape(self):
        return self.X.shape