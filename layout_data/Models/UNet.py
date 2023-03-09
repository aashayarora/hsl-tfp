import torch
import torch.nn.functional as F #?? 
from torch import nn

#Needed to reference python folders in different locations
import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import layout_data.utils.initialize as LDUI


'''
Things to do:

Remove dropout. It doesnt seem to be used in the model and was probably just a carryover from 
their testing. Need to see if its used anywhere else. 

Figure out how to import functions from other directories. The initialize_weights function 
should be imported from the utils folder instead of in here. -> Done!
'''


class _EncoderBlock(nn.Module):
    
    def __init__(model, input_channels, output_channels, dropout=False, polling=True, bn=False):
        #class initialization with inputs model, the input and output channels, and choices on whether to dropout, poll, or use batch normalization
        super(_EncoderBlock, model).__init__() #Takes initialization values from torch library
        layers = [ #Set layers as two sets of: Convolution, Normalization (Batch or Group), GELU activation
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(output_channels) if bn else nn.GroupNorm(32, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(output_channels) if bn else nn.GroupNorm(32, output_channels),
            nn.GELU(),
        ]
        
        if dropout:
            layers.append(nn.Dropout()) #Add dropout layer to the list if requested
            
        model.encode = nn.Sequential(*layers) #Set the encode atribute as the actual build of the model
        model.pool = None
        if polling:
            model.pool = nn.MaxPool2d(kernel_size=2, stride=2) #If pooling is requested, create an atribute for it
        
    def forward(model,x): #This function is called by pytorch automatically, and is the actual "running" part of the class
        if model.pool is not None:
            x = model.pool(x) #Does pooling if previously requested
        return model.encode(x) #Returns the encoded model


class _DecoderBlock(nn.Module):
    
    def __init__(model, input_channels, middle_channels, output_channels, bn=False):
        super(_DecoderBlock, model).__init__()
        layers = [ #Same layers as above. It will later be used backward to "decode"
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(output_channels) if bn else nn.GroupNorm(32, output_channels),
            nn.GELU(),
        ]
        
        #There is no pooling for the decoding steps
        
        model.decode = nn.Sequential(*layers)
        
    def forward(model, x):
        return model.decode(x)


class UNet(nn.Module):
    
    def __init__(model, classes, input_channels=3, bn=False):
        #Initializing the full UNet architecture. bn and input_channels are set, but put as inputs to make it easy to change
        super(UNet, model).__init__()
        model.enc1 = _EncoderBlock(input_channels, 64, polling=False, bn=bn)
        model.enc2 = _EncoderBlock(64, 128, bn=bn)
        model.enc3 = _EncoderBlock(128, 256, bn=bn)
        model.enc4 = _EncoderBlock(256, 512, bn=bn)
        
        model.polling = nn.AvgPool2d(kernel_size=2, stride=2)        
        model.center = _DecoderBlock(512, 1024, 512, bn=bn)
        #Pool and centering before transitioning between encoding and decoding. Similar to "flatten"
        
        model.dec4 = _DecoderBlock(1024, 512, 256, bn=bn)
        model.dec3 = _DecoderBlock(512, 256, 128, bn=bn)
        model.dec2 = _DecoderBlock(256, 128, 64, bn=bn)
        model.dec1 = nn.Sequential( #For the last convolution set, we remove padding and change the kernel size
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.GELU(),
        )
        
        model.final = nn.Conv2d(64, classes, kernel_size=1) #One last convolution
        LDUI.initialize_weights(model) #initialize weights for our model
        
    def forward(model, x): #Actually run through the model for some input
        enc1 = model.enc1(x)
        enc2 = model.enc2(enc1)
        enc3 = model.enc3(enc2)
        enc4 = model.enc4(enc3)
        center = model.center(model.polling(enc4)) #This is the "flattening" step
        dec4 = model.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False, mode='bilinear'), enc4], 1))
        dec3 = model.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False, mode='bilinear'), enc3], 1))
        dec2 = model.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False, mode='bilinear'), enc2], 1))
        dec1 = model.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False, mode='bilinear'), enc1], 1))
        #There is more required for the decoding as we need to take the previous step and merge it with the encoding output
        #It first interpolates decN to the size of encN, then concatenates the two together, then runs the decoder block
        
        final = model.final(dec1)
        
        return final    