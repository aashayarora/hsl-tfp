import numpy as np
import torch
import sys
import os
import pytorch_lightning as pl
from torch.backends import cudnn
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.utils.options import parses_ul
from layout_data.Models.model import UNetUnsupLearnSoft

def trainulsoft(hparams):
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    
    model = UNetUnsupLearnSoft(hparams)
    
    checkpoint_callback = [ModelCheckpoint(save_top_k=1, monitor='Jacobi Validation Loss', mode='min'),
			ModelCheckpoint(save_top_k=1, monitor='MAE Validation Loss', mode='min')]
    trainer = pl.Trainer(
        max_epochs = hparams.max_epochs,
        callbacks = [checkpoint_callback],
        gpus = [hparams.gpus],
        precision = 16 if hparams.use_16bit else 32,
        val_check_interval = hparams.val_check_interval,
        resume_from_checkpoint = hparams.resume_from_checkpoint,
        profiler = hparams.profiler,
        benchmark = True)
    
    print(hparams)
    print()
    trainer.fit(model)
