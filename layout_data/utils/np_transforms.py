import cv2
import torch
from torchvision import transforms

class ToTensor:
    # Transforms an input numpy array to a torch tensor
    def __init__(self, add_dim=True, type_=torch.float32):
        self.add_dim = add_dim
        self.type = type_
        
    def __call__(self, x):
        if self.add_dim:
            return torch.tensor(x, dtype=self.type).unsqueeze(0)
        return torch.tensor(x, dtype=self.type)

class Resize: #Resizes to a requested size
    def __init__(self, size):
        self.size = size
    
    def __call__(self, x):
        return cv2.resize(x, self.size)
    
class Lambda(transforms.Lambda):
    pass

class Compose(transforms.Compose): 
    pass

class Normalize(transforms.Normalize):
    pass