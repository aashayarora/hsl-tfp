import torch
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader #Loading data in the correct way
from torch.optim.lr_scheduler import ExponentialLR #Exponential learning rates
from pytorch_lightning import LightningModule #Module type to actually run the model. Need to install separately.
#pip install pytorch-lightning

import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)
    
''' Quite a few of these parts still need to be written '''
#import #Transforms
import layout_data.Models.UNet as LDMU
#import #Heatmap Visualizations 
#import #Layout Datasets
import layout_data.loss.ULLoss as LDLU

class UNetUnsupLearn(LightningModule):
    def __init__(model, hparams):
        super().__init__()
        model.hparams = hparams #Init hyper parameters 
        model.train_dataset = None #Init all the datasets, to be added to later
        model.val_dataset = None
        model.test_dataset = None
        model.build_model()
        model.build_loss()
        
    def _build_model(model):
        model.model = LDMU.UNet(input_channels=1, classes=1, bn=False) #Build the model using UNet reference
        
    def _build_loss(model):
        nx = model.hparams.nx #Set loss 
        length = model.hparams.length
        bcs = model.hparams.bcs
        model.loss = LDLU.Jacobi_layer(nx, length, bcs)
        
    def forward(model, x):
        return model.model(x)
    
    def data_loader(model, dataset, batch_size, shuffle=True):
        loader = DataLoader( #Load in the data using a specific batch size
            dataset=dataset,
            batch_size=batch_size,
            num_workers=model.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader
    
    def configure_optimizer(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams.lr) #Set to use Adam optimizer
        scheduler = ExponentialLR(optimizer, gamma=0.85) #Set to change learning rate with a "decay" of 0.85
        return [optimizer], [scheduler]
    
    def prepare_data(model): 
        #Prepare the data using the above functions
        size: int =  model.hparams.input_size #"Hint" to forces the size to be an int type
        transform_layout = transforms.Compose([ #Transform the data referencing preset mean and std layouts
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([model.hparams.mean_layout]),
                torch.tensor([model.hparams.std_layout])
            ),
        ])
        transform_heat = transforms.Compose([transforms.ToTensor()])
        
        '''
        Need to spend a bit more time fully understanding this code for training/testing/validiation
        References function from another part that is yet to be created. That should help.
        '''
        
        train_dataset = LayoutDataset(
            model.hparams.data_root, 
            subdir = model.hparams.train_dir, 
            list_path = model.hparams.train_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = model.hparams.load_name, 
            nx = model.hparams.nx)
    
        val_dataset = LayoutDataset(
            model.hparams.data_root, 
            subdir = model.hparams.val_dir, 
            list_path = model.hparams.val_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = model.hparams.load_name, 
            nx = model.hparams.nx)
        
        test_dataset = LayoutDataset(
            model.hparams.data_root, 
            subdir = model.hparams.test_dir, 
            list_path = model.hparams.test_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = model.hparams.load_name, 
            nx = model.hparams.nx)
        
        ''' Print statement was left out here ''' 

        #Assigned to model class for later use
        model.train_dataset = train_dataset
        model.val_dataset = val_dataset
        model.test_dataset = test_dataset
    
    #Call out data loading function for train, val, and test datasets to fully load the data in
    def train_dataloader(model): #Shuffle only on training to help from overtraining
        return model.data_loader(model.train_dataset, batch_size=model.hparams.batch_size)
    
    def val_dataloader(model):
        return mode.data_loader(model.val_dataset, batch_size=16, shuffle=False)
    
    def test_dataloader(model):
        return model.data_loader(model.test_dataset, batch_size=1, shuffle=False)
    
    def training_step(model, batch, batch_idx):
        layout, _ = batch
        heat_pre = model(layout)

        layout = layout * model.hparams.std_layout + model.hparams.mean_layout
        # The loss of govern equation + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = model.jacobi(layout, heat_pre, 1)

        loss_fun = LDLU.OHEMF12d(loss_fun=F.l1_loss)
        # loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.L1Loss()
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))

        loss = loss_jacobi

        model.log('loss_jacobi', loss_jacobi)
        model.log('loss', loss)

        return {"loss": loss}
    
    def validation_step(model, batch, batch_idx):
        layout, heat = batch
        heat_pre = model(layout)
        heat_pred_k = heat_pre + 298
        
        layout = layout * model.hparams.std_layout + model.hparams.mean_layout
        
        loss_jacobic = F.l1loss(heat_pre, model.jacobi(layout, heat_pre.detach(), 1))
        
        val_mae = F.l1_loss(heat_pred_k, heat)
        
        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_ics, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pre_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, model.hparams.nx)
            y = np.linspace(0, 0.1, model.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list, model.current_epoch)
        
        return {"Jacobi Validation Loss": loss_jacobi, "MAE Validation Loss": val_mae}
    
    def validation_epoch_end(model, outputs):
        val_loss_jacobi_mean = torch.stack([x["Jacobi Validation Loss"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["MAE Validation Loss"] for x in outputs]).mean()
        
        model.log("Jacobi Validation Loss", val_loss_jacobi_mean)
        model.log("MAE Validation Loss", val_mae_mean)