from pathlib import Path
import numpy as np
import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.Models.model import UNet_SupLearn

def trainsl(hparams):
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = UNet_SupLearn(hparams)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_mae_mean', mode='min')
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gpus=[hparams.gpus],
        precision=16 if hparams.use_16bit else 32,
        val_check_interval = hparams.val_check_interval,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
        benchmark=True)
    
    print(hparams)
    print()
    trainer.fit(model)