from typing import Optional
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from .generic import WindowDataset, data_path


class TRAJ(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024, window_size: int = 10, shuffle: bool = True):
        super().__init__()
        if window_size not in [10, 20, 30]:
            raise ValueError("window size have to be in [10, 20, 30]")
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.prepare_data_per_node = True
        self.num_variables = window_size * 2
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = WindowDataset(
                data_path / f"STR/train_{self.window_size}.csv",
                target_column='label', index_col='id', serie_col='travel'
            )

        if stage == "predict" or stage is None:
            self.test_set = WindowDataset(
                data_path / f"STR/test_{self.window_size}.csv",
                target_column='label', index_col='id', serie_col='travel'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=6
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=6
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=6
        )