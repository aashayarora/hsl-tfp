# tf packages for conversion
import tensorflow as tf

# torch files for data loading
import torch
import layout_data.data.layout as LDDA
import layout_data.utils.np_transforms as LDUNP

class DataLoader():

    def __init__(self, hparams) -> None:
        self.hparams = hparams

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.prepare_data()

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

        '''
        Need to spend a bit more time fully understanding this code for training/testing/validiation
        References function from another part that is yet to be created. That should help.
        '''

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

        # ''' Print statement was left out here ''' 
        # print(type(train_dataset))
        # print(train_dataset)

        tf_train_dataset = tf.convert_to_tensor(train_dataset.numpy())
        tf_val_dataset = tf.convert_to_tensor(val_dataset.numpy())
        tf_test_dataset = tf.convert_to_tensor(test_dataset.numpy())

        # Assigned to model class for later use
        self.train_dataset = tf_train_dataset
        self.val_dataset = tf_val_dataset
        self.test_dataset = tf_test_dataset