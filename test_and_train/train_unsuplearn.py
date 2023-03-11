import numpy as np
import torch
import sys
import os
import pytorch_lightning as pl
from torch.backends import cudnn
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

#Gets your current directory and adds it to the system paths
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.utils.options import parses_ul
from layout_data.Models.model import UNetUnsupLearn

def main(hparams):
    #The actual training routine
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    
    #initialize model
    model = UNetUnsupLearn(hparams)
    
    #initialize training
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='Jacobi Validation Loss', mode='min')
    trainer = pl.Trainer(
        max_epochs = hparams.max_epochs,
        callbacks = [checkpoint_callback],
        gpus = [hparams.gpus],
        precision = 16 if hparams.use_16bit else 32,
        val_check_interval = hparams.val_check_interval,
        resume_from_checkpoint = hparams.resume_from_checkpoint,
        profiler = hparams.profiler,
        weights_summary = None,
        benchmark = True)
    
    #start training
    print(hparams)
    print()
    trainer.fit(model)

if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)
    main(hparams)