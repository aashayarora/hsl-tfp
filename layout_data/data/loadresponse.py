import os
import numpy as np
import scipy.io as sio
import h5py
from torchvision.datasets import VisionDataset

class LoadResponse(VisionDataset):
    # Load in information of the dataset
    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name='F',
        resp_name='u',
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        max_iters=None,
        nx=200):
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.nx = nx
        self.extensions = extensions
        self.sample_files = make_dataset_list(root, list_path, extensions, is_valid_file, max_iters=max_iters)
        
    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name, self.nx)
        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp
    
    def __len__(self):
        return len(self.sample_files)
    
def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None, max_iters=None):
    # make_dataset() function from torchvision
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)): #^ = XOR
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)
        
    #assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf: #Finds all the files in the list_path directory that was passed in
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    if max_iters is not None:
        files = files * int(np.ceil(float(max_iters) / len(files)))
        files = files[:max_iters]
    return files

def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def mat_loader(path, load_name, resp_name=None, nx=200):
    #Loads in the layout of the data from matlab files
    mats = sio.loadmat(path)
    if load_name == 'F':
        load = mats.get(load_name).astype(np.float32)
    elif load_name == 'list':
        load = mats.get(load_name)[0]
        layout_map = np.zeros((nx, nx))
        mul = int(nx/10)
        for i in load:
            i = i-1
            layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones((mul, mul))
        load = layout_map
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp