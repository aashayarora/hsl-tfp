import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LayerNormalization, ReLU, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, AveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

#Needed to reference python folders in different locations
import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import layout_data.utils.initialize as LDUI

#Needed to reference python folders in different locations
import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import layout_data.utils.initialize as LDUI

class _EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, input_channels, output_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__() 
        layers = [
            Conv2D(output_channels, kernel_size=3, padding="same", activation=None),
            BatchNormalization() if bn else LayerNormalization(),
            ReLU(),
            Conv2D(output_channels, kernel_size=3, padding="same", activation=None),
            BatchNormalization() if bn else LayerNormalization(),
            ReLU(),
        ]
        self.encode = Sequential()
        for layer in layers:
            self.encode.add(layer)

        self.dropout = Dropout(rate=0.5) if dropout else None

        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2) if polling else None

    def call(self, x):
        x = self.encode(x)
        if self.dropout:
            x = self.dropout(x)
        if self.pool:
            x = self.pool(x)
        return x

class _DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, input_channels, middle_channels, output_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        layers = [
            Conv2D(middle_channels, kernel_size=3, padding="same", activation=None),
            BatchNormalization() if bn else LayerNormalization(),
            ReLU(),
            Conv2D(output_channels, kernel_size=3, padding="same", activation=None),
            BatchNormalization() if bn else LayerNormalization(),
            ReLU(),
        ]
        self.decode = Sequential()
        for layer in layers:
            self.decode.add(layer)

    def call(self, x):
        return self.decode(x)

class UNet(tf.keras.Model):
    
    def __init__(self, classes, input_channels=3, bn=False):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(input_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(64, 128, bn=bn)
        self.enc3 = _EncoderBlock(128, 256, bn=bn)
        self.enc4 = _EncoderBlock(256, 512, bn=bn)
        
        self.polling = AveragePooling2D(pool_size=(2, 2), strides=2)        
        self.center = _DecoderBlock(512, 1024, 512, bn=bn)
        
        self.dec4 = _DecoderBlock(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlock(256, 128, 64, bn=bn)
        layers = [
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization() if bn else LayerNormalization(),
            Conv2D(64, kernel_size=1, padding='same', activation='relu'),
            BatchNormalization() if bn else LayerNormalization(),
        ]
        self.dec1 = Sequential()
        for layer in layers:
            self.dec1.add(layer)
        
        self.final = Conv2D(classes, kernel_size=1)
        LDUI.initialize_weights(self)
        
    def call(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(tf.concat([tf.image.resize(center, enc4.shape[1:3]), enc4], axis=-1))
        dec3 = self.dec3(tf.concat([tf.image.resize(dec4, enc3.shape[1:3]), enc3], axis=-1))
        dec2 = self.dec2(tf.concat([tf.image.resize(dec3, enc2.shape[1:3]), enc2], axis=-1))
        dec1 = self.dec1(tf.concat([tf.image.resize(dec2, enc1.shape[1:3]), enc1], axis=-1))
        final = self.final(dec1)
        return final
