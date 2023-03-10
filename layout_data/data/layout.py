import os
from .loadresponse import LoadResponse, mat_loader

class LayoutDataset(LoadResponse):
    # Layout the datasets generated by the layout generator
    def __init__(self, root, list_path, subdir, transform=None, target_transform=None, load_name='list', 
                 resp_name='u', max_iters=None, nx=200):
        root = os.path.join(root, subdir)
        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions='mat',
            transform=transform
            target_transform=target_transform
            max_iters=max_iters
            nx=nx)
        #Running this class essentially just consolidates a couple other classes into a single import so its easier to use. Nothing extra. 
        