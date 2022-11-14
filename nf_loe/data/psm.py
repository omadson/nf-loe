from pathlib import Path
from typing import Optional
from nf_loe.data import GenericDataset
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from .generic import create_window_df, data_path


class PSM(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024, window_size: int = 1, shuffle: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.index_col = "timestamp_(min)"
        self.shuffle = shuffle
        self.num_variables = pd.read_csv(data_path / "PSM/train.csv", nrows=1).shape[1] -1
        self.prepare_data_per_node = True
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = GenericDataset(
                data_path / "PSM/train.csv",
                index_col="timestamp_(min)",
                window_size=self.window_size
            )

        if stage == "predict" or stage is None:
            self.test_labels = pd.read_csv(
                data_path / 'PSM/test_label.csv',
                index_col="timestamp_(min)"
            )
            self.test_set = pd.read_csv(
                data_path / 'PSM/test.csv',
                index_col="timestamp_(min)"
            )
            df_test = self.test_set.join(self.test_labels)

            self.test_set = GenericDataset(
                data_frame=df_test, 
                index_col="timestamp_(min)", 
                target_column="label",
                scaler=self.train_set.scaler
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
            batch_size=self.test_set[:]['sample'].shape[0], 
            num_workers=6
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.test_set[:]['sample'].shape[0], 
            num_workers=6
        )