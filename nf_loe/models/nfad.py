import torch
from torch.distributions import MultivariateNormal
import pandas as pd

import wandb
import torchmetrics
import pytorch_lightning as pl

from .flows import RealNVP, MAF, FlowModule


# class NFAD(pl.LightningModule):
#     def __init__(self, flow=RealNVP, classifier=, p=)