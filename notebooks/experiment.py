from enum import Enum
from time import time_ns
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import torch
from sklearn.metrics import roc_curve

from torch import nn
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from nf_loe.data import SyntheticDataset, GenericDataset
from torch.utils.data import DataLoader
import pandas as pd
import typer
from nflows.flows.realnvp import SimpleRealNVP
from nflows.flows.autoregressive import MaskedAutoregressiveFlow

import mlflow.pytorch
from mlflow import MlflowClient

import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger


mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

class MAF(pl.LightningModule):
    def __init__(self, num_variables, num_flows, lr, weight_decay, loss_wrapper=None):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_wrapper = loss_wrapper
        self.model = MaskedAutoregressiveFlow(
            features=num_variables,
            hidden_features=num_variables*2,
            num_layers=num_flows,
            num_blocks_per_layer=2,
            batch_norm_between_layers=True
        )
    def forward(self, x):
        return self.model.log_prob(inputs=x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        ) 
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X = train_batch["sample"]
        if self.loss_wrapper != None:
            loss = self.loss_wrapper(self.forward(X))
        else:
            loss = self.forward(X).mean()
        self.logger.experiment.whatever_ml_flow_supports(...)
        return {"loss": loss}
    
    def test_step(self, test_batch, batch_idx):
        X = test_batch["sample"]
        y_hat = self.forward(X)
        fpr, tpr, thresholds = roc_curve(test_batch["label"].detach().numpy(), y_hat.detach().numpy())
        print(pr, tpr, thresholds)
        self.logger.experiment.whatever_ml_flow_supports(...)
        metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "y_hat": y_hat
        }
        self.logger.log_metrics(
            metrics
        )
        return metrics




def RealNVP(num_variables, num_flows, actnorm):
    return SimpleRealNVP(
        features=num_variables,
        hidden_features=num_variables*2,
        num_layers=num_flows,
        num_blocks_per_layer=2,
        batch_norm_between_layers=True
    )

class Model(str, Enum):
    realnvp = "RealNVP"
    maf = "MAF"

models = {
    "RealNVP": RealNVP,
    # "NICE": NICE,
    "MAF": MAF,
    # "Glow": Glow,
    # "NeuralSpline": NeuralSpline,
}

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def main(
    flow: Model=Model.realnvp,
    num_flows:int=10,
    actnorm: bool=False,
    train_method:str='blind',
    contamination:float=.1,
    max_epochs:int=200,
    log_frequency:int=10,
    batch_size:int=1024,
    lr:float=1e-4,
    weight_decay:float=1e-5,
    data:str='psm',
    window_size:int=1,
):
    setup_args = locals()
    print(title:='SETUP')
    print('=' * len(title))
    for key, value in setup_args.items():
        print(f' - {key}: {value}')
    print()
    # load data
    print('loading data...', end=' ')
    if data == 'psm':
        train_set, test_set, train_loader, test_loader, test_labels = load_psm_data(batch_size, window_size)
    print('data loaded.')
    # build flow
    print('building model...', end=' ')
    num_variables = train_set[1]['sample'].shape[0]
    # model = models[flow](num_variables, num_flows, actnorm)
    model = models[flow](
        num_variables=num_variables,
        num_flows=num_flows,
        lr=lr,
        weight_decay=weight_decay,
        loss_wrapper=None
    )
    # training
    trainer = pl.Trainer(
        accelerator='CPU',
        limit_train_batches=0.5,
        max_epochs=max_epochs,
        logger=mlf_logger
    )
    trainer.fit(model, train_loader)
    # mlflow.pytorch.autolog()
    # with mlflow.start_run() as run:
    #     trainer.fit(model, train_loader)
    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


def simple_loss(loss):
    loss_n = -loss
    loss_a = -torch.log(1 - torch.exp(loss_n * (-1)))
    return loss_n, loss_a

def vanilla_loss(loss):
    loss_n = -loss
    loss_a = 0
    return loss_n, loss_a


def load_psm_data(batch_size, window_size=1):
    # train
    train_set = GenericDataset(
        "../data/PSM/train.csv",
        index_col="timestamp_(min)",
        window_size=window_size
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3
    )
    # test
    test_labels = pd.read_csv('../data/PSM/test_label.csv', index_col="timestamp_(min)")
    test_set = pd.read_csv('../data/PSM/test.csv', index_col="timestamp_(min)")
    df_test = test_set.join(test_labels)

    test_set = GenericDataset(
        data_frame=df_test, 
        index_col="timestamp_(min)", 
        target_column="label"
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=3
    )
    test_labels = pd.read_csv('../data/PSM/test_label.csv')['label'].values
    return train_set, test_set, train_loader, test_loader, test_labels


if __name__ == "__main__":
    typer.run(main)