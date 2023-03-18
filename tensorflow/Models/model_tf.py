import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import tensorflow as tf

import tensorflow.Models.UNet as UNet

# Fully written in torch - Must be ported
import tensorflow.loss.ULLoss as LDLU

class UNetUnsupLearn():

    def __init__(self, hparams):
        self.hparams = hparams

        self.model = None
        self._build_model()

        self.loss = None
        self._build_loss()

    def training_step(self, batch, batch_idx):
        layout, _ = batch
        heat_pre = self(layout)

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout
        # The loss of govern equation + Online Hard Sample Mining
        with tf.stop_gradient():
            heat_jacobi = self.loss(layout, heat_pre, 1)

        # Need to change this to tf version, unsure how
        loss_fun = LDLU.OHEMF12d(loss_fun=F.l1_loss)
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, tf.zeros_like(heat_pre - heat_jacobi))

        loss = loss_jacobi

        # self.log('loss_jacobi', loss_jacobi)
        # self.log('loss', loss)

        return {"loss": loss}

    def _build_model(self):
        self.model = UNet.UNet(classes=1, input_channels=1, bn=False) #Build the model using UNet reference   

    def _build_loss(self):
        nx = self.hparams.nx #Set loss 
        length = self.hparams.length
        bcs = self.hparams.bcs
        self.loss = LDLU.Jacobi_layer(nx, length, bcs)

    