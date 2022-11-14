from enum import Enum

import pandas as pd
import typer
from rich import print
from rich.table import Table
from rich.console import Console
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from nf_loe.models import RealNVP, MAF
from nf_loe.data import PSM, SMAP, TRAJ


class Model(str, Enum):
    realnvp = "RealNVP"
    maf = "MAF"

class Data(str, Enum):
    psm = "PSM"
    smap = "SMAP"
    traj = "TRAJ"


pl.seed_everything(42)

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    flow: Model=Model.realnvp,
    num_flows:int=10,
    train_method:str='blind',
    contamination:float=.1,
    max_epochs:int=200,
    batch_size:int=1024,
    lr:float=1e-4,
    weight_decay:float=1e-5,
    dataset:Data=Data.psm,
    window_size:int=1,
    accelerator="cpu"
):
    setup_args = locals()
    wandb_logger = WandbLogger()
    console = Console()
    table = Table(title='Setup parameters')
    table.add_column("Parameter")
    table.add_column("Value")
    for key, value in setup_args.items():
        table.add_row(key, str(value))
    console.print(table)
    # load data
    with console.status("loading data..."):
        data = eval(dataset)(window_size=window_size)
    print('data loaded!')
    with console.status("building model..."):
        model = eval(flow)(
            num_variables=data.num_variables,
            num_flows=num_flows,
            lr=lr,
            weight_decay=weight_decay,
            loss_wrapper=None
        )
    print('built model!')
    # training
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=wandb_logger,
    )
    trainer.fit(model, data)
    results = trainer.predict(model, data)
    # save first results
    df = pd.DataFrame({
        "y_hat": results[0][0].ravel(),
        "y_true": results[0][1].ravel(),
        "serie": results[0][2].ravel(),
        "point": results[0][3].ravel()
    })
    df.to_csv('results.csv', index=False)
    # save remaing
    for result in results[1:]:
        df = pd.DataFrame({
         "y_hat": result[0].ravel(),
         "y_true": result[1].ravel(),
         "serie": result[2].ravel(),
         "point": result[3].ravel()
        })
        df.to_csv('results.csv', mode='a', index=False, header=False)
    # trainer.test(model, data)

if __name__ == "__main__":
# foo = main(max_epochs=1, flow='RealNVP', num_flows=2, dataset='TRAJ', window_size=10)
    app()