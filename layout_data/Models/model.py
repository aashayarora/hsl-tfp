import torch
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader # Loading data in the correct way
from torch.optim.lr_scheduler import ExponentialLR # Exponential learning rates
from pytorch_lightning import LightningModule # Module type to actually run the model. Need to install separately.
# pip install pytorch-lightning

import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import layout_data.utils.np_transforms as LDUNP
import layout_data.Models.UNet as UNet
import layout_data.utils.visualize as LDUV
import layout_data.data.layout as LDDA
import layout_data.loss.ULLoss as LDLU

class UNetUnsupLearn(LightningModule):
    #Unssupervised Learning using the UNet architecture
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams) # Init hyper parameters (different function needed for hyper parameters)

        self.train_dataset = None # Init all the datasets, to be added to later
        self.val_dataset = None
        self.test_dataset = None

        self.model = None
        self._build_model()

        self.loss = None
        self._build_loss()
        
    def _build_model(self):
        self.model = UNet.UNet(input_channels=1, classes=1, bn=False) #Build the model using UNet reference
        
    def _build_loss(self):
        #Builds the loss function from the hyperparameters
        nx = self.hparams.nx 
        length = self.hparams.length
        bcs = self.hparams.bcs
        self.loss = LDLU.Jacobi_layer(nx, length, bcs) #Using Jacobian loss
        
    def forward(self, x):
        return self.model(x) #Actually run the model (forward class automatically run in torch)
    
    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader( #Load in the data using a specific batch size
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # Set to use Adam optimizer
        scheduler = ExponentialLR(optimizer, gamma=0.85) # Set to change learning rate with a "decay" of 0.85
        return [optimizer], [scheduler]
    
    def prepare_data(self): 
        # Prepare the data using the above functions
        size: int =  self.hparams.input_size # "Hint" to forces the size to be an int type
        transform_layout = LDUNP.Compose([ # Transform the data referencing preset mean and std layouts
            LDUNP.ToTensor(),
            LDUNP.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout])
            ),
        ])
        transform_heat = LDUNP.Compose([LDUNP.ToTensor()])
        
        #Build the 3 different datasets
        train_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.train_dir, 
            list_path = self.hparams.train_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
    
        val_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.val_dir, 
            list_path = self.hparams.val_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
        
        test_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.test_dir, 
            list_path = self.hparams.test_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
        
        # Assigned to model class for later use
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    # Call out data loading function for train, val, and test datasets to fully load the data in
    def train_dataloader(self): # Shuffle only on training to help from overtraining
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self): #Load in validation data
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)
    
    def test_dataloader(self): #Load in test data
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)
    
    def training_step(self, batch, batch_idx): #A single step of training
        layout, _ = batch
        heat_pre = self(layout) 

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout #Normalization
        # The loss of govern equation + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.loss(layout, heat_pre, 1)

        loss_fun = LDLU.OHEMF12d(loss_fun=F.l1_loss)
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi)) 

        loss = loss_jacobi

        self.log('loss_jacobi', loss_jacobi)
        self.log('loss', loss)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx): #A step of validation
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298 #Use kelvin instead of celsius
        
        layout = layout * self.hparams.std_layout + self.hparams.mean_layout #Normalization
        
        loss_jacobi = F.l1_loss(heat_pre, self.loss(layout, heat_pre.detach(), 1)) #Jacobian loss
        
        val_mae = F.l1_loss(heat_pred_k, heat) #MAE Loss
        
        if batch_idx == 0: #On the first batch index of validation (at the end of each training epoch), create a heatmap
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            LDUV.visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)
        
        return {"Jacobi Validation Loss": loss_jacobi, "MAE Validation Loss": val_mae}
    
    def validation_epoch_end(self, outputs): #At the end of the each validation epoch, log the MAE and Jacobian losses
        val_loss_jacobi_mean = torch.stack([x["Jacobi Validation Loss"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["MAE Validation Loss"] for x in outputs]).mean()
        
        self.log("Jacobi Validation Loss", val_loss_jacobi_mean)
        self.log("MAE Validation Loss", val_mae_mean)
        
    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
    
    
class UNetSupLearn(LightningModule):
    #Supervised learning, for comparison against unsupervised learning
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self._build_model()
        
    def _build_model(self): #As before, use our UNet architecture
        self.model = UNet.UNet(input_channels = 1, classes = 1, bn=False)
    
    def forward(self, x): #Create model
        return self.model(x)
    
    def __dataloader(self, dataset, batch_size, shuffle=True): #Load data
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers = self.hparams.num_workers,
            shuffle=shuffle)
        return loader
    
    def configure_optimizers(self): #Same optimizers and learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer],[scheduler]
    
    def prepare_data(self): #Prepare the data, same as before
        size: int = self.hparams.input_size
        transform_layout = LDUNP.Compose([
            LDUNP.ToTensor(),
            LDUNP.Normalize(torch.tensor([self.hparams.mean_layout]), torch.tensor([self.hparams.std_layout]))])
        
        transform_heat = LDUNP.Compose([LDUNP.ToTensor()])
        
        train_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat, load_name=self.hparams.load_name,
            nx=self.hparams.nx)
        val_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat, load_name=self.hparams.load_name,
            nx=self.hparams.nx)
        test_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat, load_name=self.hparams.load_name,
            nx=self.hparams.nx)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
    def train_dataloader(self): #Load data, same as before
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)
    
    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)
    
    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        
        loss_fun = torch.nn.L1Loss() #Use different loss function - supervised learning
        loss = loss_fun(heat_pre, heat - 298.0) 
        
        self.log('loss',loss)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx): #Same validation steps, including creation of heatmap
        layout, heat = batch
        heat_pre = self(layout)
        heat_pre_k = heat_pre + 298
        
        val_mae = F.l1_loss(heat_pre_k, heat)
        
        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pre_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            LDUV.visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)
            
        return {"val_mae": val_mae}
    
    def validation_epoch_end(self, outputs): #Log just MAE loss, as we dont have Jacobian loss in supervised learning
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()
        self.log('val_mae_mean', val_mae_mean)
        
    def test_step(self, batch, batch_idx):
        pass
    
    def test_epoch_end(self, outputs):
        pass

class UNetUnsupLearnSoft(LightningModule):
#Unsupervised learning with soft constraints on the Neumann boundary   
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.model = None
        self._build_model()

        self.loss = None
        self._build_loss()
        
    def _build_model(self):
        self.model = UNet.UNet(input_channels=1, classes=1, bn=False)
        
    def _build_loss(self):
        nx = self.hparams.nx 
        length = self.hparams.length
        bcs = self.hparams.bcs
        self.loss = LDLU.Jacobi_layerSoft(nx, length, bcs)
        
    def forward(self, x):
        return self.model(x)
    
    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]
    
    def prepare_data(self): 

        size: int =  self.hparams.input_size
        transform_layout = LDUNP.Compose([
            LDUNP.ToTensor(),
            LDUNP.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout])
            ),
        ])
        transform_heat = LDUNP.Compose([LDUNP.ToTensor()])
        
        train_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.train_dir, 
            list_path = self.hparams.train_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
    
        val_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.val_dir, 
            list_path = self.hparams.val_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
        
        test_dataset = LDDA.LayoutDataset(
            self.hparams.data_root, 
            subdir = self.hparams.test_dir, 
            list_path = self.hparams.test_list,
            transform = transform_layout, 
            target_transform = transform_heat,
            load_name = self.hparams.load_name, 
            nx = self.hparams.nx)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)
    
    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)
    
    def training_step(self, batch, batch_idx):
        layout, _ = batch
        heat_pre = self(layout) 

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout

        with torch.no_grad():
            heat_jacobi = self.loss(layout, heat_pre, 1)

        loss_fun = LDLU.OHEMF12d(loss_fun=F.l1_loss)
        # loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.L1Loss()
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))
        loss_D = F.l1_loss(heat_pre[..., 90:110, :1], torch.zeros_like(heat_pre[..., 90:110, :1]))

        loss = loss_jacobi + 0.001*loss_D #Combined physics loss and data loss

        self.log('loss_jacobi', loss_jacobi)
        self.log('loss_D', loss_D)
        self.log('loss', loss)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298
        
        layout = layout * self.hparams.std_layout + self.hparams.mean_layout
        
        loss_jacobi = F.l1_loss(heat_pre, self.loss(layout, heat_pre.detach(), 1))
        
        val_mae = F.l1_loss(heat_pred_k, heat)
        
        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            LDUV.visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)
        
        return {"Jacobi Validation Loss": loss_jacobi, "MAE Validation Loss": val_mae}
    
    def validation_epoch_end(self, outputs):
        val_loss_jacobi_mean = torch.stack([x["Jacobi Validation Loss"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["MAE Validation Loss"] for x in outputs]).mean()
        
        self.log("Jacobi Validation Loss", val_loss_jacobi_mean)
        self.log("MAE Validation Loss", val_mae_mean)
        
    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
