import torch
import pytorch_lightning as pl
from torch.distributions import MultivariateNormal
from nflows.flows.realnvp import SimpleRealNVP
from nflows.flows.autoregressive import MaskedAutoregressiveFlow




class FlowModule(pl.LightningModule):
    def forward(self, x):
        return self.flow.log_prob(inputs=x)
    
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
            loss = -self.forward(X).mean()
        self.log("loss", loss)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        X = batch["sample"]
        if self.predict_wrapper != None:
            return self.predict_wrapper(X), batch["label"], batch["serie"], batch["point"]
        return -self.forward(X), batch["label"], batch["serie"], batch["point"]

    def test_step(self, test_batch, batch_idx):
        y_hat = self.predict_step(test_batch, batch_idx)
        y_true = test_batch["label"].T[0].int()
        # results = {
        #     "y_hat": y_hat.tolist(),
        #     "y_true": y_true.tolist(),
        # }
        # pd.DataFrame(results).to_csv('metrics.csv', index=False)


class MAF(FlowModule):
    def __init__(self, num_variables, num_flows, lr, weight_decay, loss_wrapper=None, predict_wrapper=None):
        FlowModule.__init__(self)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_wrapper = loss_wrapper
        self.predict_wrapper = predict_wrapper
        self.flow = MaskedAutoregressiveFlow(
            features=num_variables,
            hidden_features=256,
            num_layers=num_flows,
            num_blocks_per_layer=3,
            batch_norm_between_layers=True
        )
    



class RealNVP(FlowModule):
    def __init__(self, num_variables, num_flows, lr, weight_decay, loss_wrapper=None, predict_wrapper=None):
        FlowModule.__init__(self)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_wrapper = loss_wrapper
        self.predict_wrapper = predict_wrapper
        self.flow = SimpleRealNVP(
            features=num_variables,
            hidden_features=num_variables*2,
            num_layers=num_flows,
            num_blocks_per_layer=2,
            batch_norm_between_layers=True
        )
