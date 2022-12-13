import numpy as np, time

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, Activation, Dropout, concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import ConvLSTM2D #, ConvLSTM3D only from tf v2.6.0
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model

def Unet(img_shape, coarse_dim, ks=3, dropout=0.05, path='./'):
    # print message at runtime
    if(img_shape[0] == 64 and np.size(img_shape) == 3):
        print('Create 2D U-Net network with 3 levels...\n')
    elif(img_shape[0] == 128 and np.size(img_shape) == 3):
        print('Create 2D U-Net network with 4 levels...\n')
    elif(img_shape[0] == 64 and np.size(img_shape) == 4):
        print('Create 3D U-Net network with 3 levels...\n')
    elif(img_shape[0] == 128  and np.size(img_shape) == 4):
        print('Create 3D U-Net network with 4 levels...\n')
    else:
        print('???')
    
    def Conv2D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A2' %layer_name)(a)
        return a


    def Conv3D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A2' %layer_name)(a)
        return a

    
    img_input = Input(shape=img_shape, name='Image')
        
    # U-Net Encoder - upper level
    if(np.size(img_shape) == 3):
        # 2-D network
        e1c = Conv2D_Layers(prev_layer=img_input, nr_filts=int(coarse_dim/16),
                            kernel_size=ks, layer_name='E1')
        e1 = MaxPooling2D(pool_size=(2, 2), name='E1_P')(e1c)
        e1 = Dropout(dropout*0.5, name='E1_D2')(e1)
    elif(np.size(img_shape) == 4):
        # 3-D network
        e1c = Conv3D_Layers(prev_layer=img_input, nr_filts=int(coarse_dim/16),
                            kernel_size=(ks, ks, ks), layer_name='E1')
        e1 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_P')(e1c)
        e1 = Dropout(dropout*0.5, name='E1_D2')(e1)

    # U-Net Encoder - second level
    if(np.size(img_shape) == 3):
        # 2-D network
        e2c = Conv2D_Layers(prev_layer=e1, nr_filts=int(coarse_dim/8),
                            kernel_size=ks, layer_name='E2')
        e2 = MaxPooling2D(pool_size=(2, 2), name='E2_P')(e2c)
        e2 = Dropout(dropout, name='E2_D2')(e2)
    elif(np.size(img_shape) == 4):
        # 3-D network
        e2c = Conv3D_Layers(prev_layer=e1, nr_filts=int(coarse_dim/8),
                            kernel_size=(ks, ks, ks), layer_name='E2')
        e2 = MaxPooling3D(pool_size=(2, 2, 2), name='E2_P')(e2c)
        e2 = Dropout(dropout, name='E2_D2')(e2)

    # U-Net Encoder - third level
    if(np.size(img_shape) == 3):
        # 2-D network
        e3c = Conv2D_Layers(prev_layer=e2, nr_filts=int(coarse_dim/4),
                            kernel_size=ks, layer_name='E3')
        e3 = MaxPooling2D(pool_size=(2, 2), name='E3_P')(e3c)
        e3 = Dropout(dropout, name='E3_D2')(e3) 
    elif(np.size(img_shape) == 4):
        # 3-D network
        e3c = Conv3D_Layers(prev_layer=e2, nr_filts=int(coarse_dim/4),
                            kernel_size=(ks, ks, ks), layer_name='E3')
        e3 = MaxPooling3D(pool_size=(2, 2, 2), name='E3_P')(e3c)
        e3 = Dropout(dropout, name='E3_D2')(e3)  

    if(img_shape[0] >= 64 and img_shape[0] < 128):
        # U-Net Encoder - bottom level
        if(np.size(img_shape) == 3):
            # 2-D network
            b = Conv2D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2), kernel_size=(ks, ks), layer_name='B')
            
            d3 = Conv2DTranspose(filters=int(coarse_dim/4), kernel_size=(ks, ks), 
                                strides=(2, 2), padding='same', name='D3_DC')(b)
        elif(np.size(img_shape) == 4):
            # 3-D network
            b = Conv3D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2), kernel_size=(ks, ks, ks), layer_name='B')
            
            d3 = Conv3DTranspose(filters=int(coarse_dim/4), kernel_size=(ks, ks, ks), 
                                strides=(2, 2, 2), padding='same', name='D3_DC')(b)
    elif(img_shape[0] >= 128):
        if(np.size(img_shape) == 3):
            # 2-D network
            # U-Net Encoder - fourth level
            e4c = Conv2D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2),
                                kernel_size=ks, layer_name='E4')
            e4 = MaxPooling2D(pool_size=(2, 2), name='E4_P')(e4c)
            e4 = Dropout(dropout, name='E4_D2')(e4)  
                
            # U-Net Encoder - bottom level
            b = Conv2D_Layers(prev_layer=e4, nr_filts=coarse_dim, kernel_size=ks, layer_name='B')

            # U-Net Decoder - fourth level
            d4 = Conv2DTranspose(filters=int(coarse_dim/2), kernel_size=ks, 
                                strides=(2, 2), padding='same', name='D4_DC')(b)
            d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
            d4 = Dropout(dropout, name='D4_D1')(d4)
            d4 = Conv2D_Layers(prev_layer=d4, nr_filts=int(coarse_dim/2), 
                            kernel_size=(ks, ks), layer_name='D4')

            # U-Net Decoder - third level
            d3 = Conv2DTranspose(filters=int(coarse_dim/4), kernel_size=ks, 
                                strides=(2, 2), padding='same', name='D3_DC')(d4)
        elif(np.size(img_shape) == 4):
            # 3-D network
            # U-Net Encoder - fourth level
            e4c = Conv3D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2),
                                kernel_size=(ks, ks, ks), layer_name='E4')
            e4 = MaxPooling3D(pool_size=(2, 2, 2), name='E4_P')(e4c)
            e4 = Dropout(dropout, name='E4_D2')(e4)  
                
            # U-Net Encoder - bottom level
            b = Conv3D_Layers(prev_layer=e4, nr_filts=coarse_dim, kernel_size=(ks, ks, ks), layer_name='B')

            # U-Net Decoder - fourth level
            d4 = Conv3DTranspose(filters=int(coarse_dim/2), kernel_size=(ks, ks, ks), 
                                strides=(2, 2, 2), padding='same', name='D4_DC')(b)
            d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
            d4 = Dropout(dropout, name='D4_D1')(d4)
            d4 = Conv3D_Layers(prev_layer=d4, nr_filts=int(coarse_dim/2), 
                            kernel_size=(ks, ks, ks), layer_name='D4')
            
            # U-Net Decoder - third level
            d3 = Conv3DTranspose(filters=int(coarse_dim/4), kernel_size=(ks, ks, ks), 
                            strides=(2, 2, 2), padding='same', name='D3_DC')(d4)
    else:
        print('ERROR: input data have wrong dimension')

    # U-Net Decoder - third level (continue)
    if(np.size(img_shape) == 3):
        # 2-D network
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(dropout, name='D3_D1')(d3)
        d3 = Conv2D_Layers(prev_layer=d3, nr_filts=int(coarse_dim/2), 
                           kernel_size=(ks, ks), layer_name='D3')
    elif(np.size(img_shape) == 4):
        # 3-D network
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(dropout, name='D3_D1')(d3)
        d3 = Conv3D_Layers(prev_layer=d3, nr_filts=int(coarse_dim/2), 
                           kernel_size=(ks, ks, ks), layer_name='D3')

    # U-Net Decoder - second level
    if(np.size(img_shape) == 3):
        # 2-D network
        d2 = Conv2DTranspose(filters=int(coarse_dim/8), kernel_size=ks, 
                        strides=(2, 2), padding='same', name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
        d2 = Dropout(dropout, name='D2_D1')(d2)
        d2 = Conv2D_Layers(prev_layer=d2, nr_filts=int(coarse_dim/4),
                    kernel_size=(ks, ks), layer_name='D2')
    elif(np.size(img_shape) == 4):
        # 3-D network
        d2 = Conv3DTranspose(filters=int(coarse_dim/8), kernel_size=(ks, ks, ks), 
                            strides=(2, 2, 2), padding='same', name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
        d2 = Dropout(dropout, name='D2_D1')(d2)
        d2 = Conv3D_Layers(prev_layer=d2, nr_filts=int(coarse_dim/4),
                           kernel_size=(ks, ks, ks), layer_name='D2')

    # U-Net Decoder - upper level
    if(np.size(img_shape) == 3):
        d1 = Conv2DTranspose(filters=int(coarse_dim/16), kernel_size=ks, 
                            strides=(2, 2), padding='same', name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
        d1 = Dropout(dropout, name='D1_D1')(d1)
        d1 = Conv2D_Layers(prev_layer=d1, nr_filts=int(coarse_dim/16),
                        kernel_size=(ks, ks), layer_name='D1')
    elif(np.size(img_shape) == 4):
        d1 = Conv3DTranspose(filters=int(coarse_dim/16), kernel_size=(ks, ks, ks), 
                            strides=(2, 2, 2), padding='same', name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
        d1 = Dropout(dropout, name='D1_D1')(d1)
        d1 = Conv3D_Layers(prev_layer=d1, nr_filts=int(coarse_dim/16),
                        kernel_size=(ks, ks, ks), layer_name='D1')

    # Outro Layer
    if(np.size(img_shape) == 3):
        output_image = Conv2D(filters=int(img_shape[-1]), kernel_size=ks, 
                              strides=(1, 1), padding='same', name='out_C')(d1)
    elif(np.size(img_shape) == 4):
        output_image = Conv3D(filters=int(img_shape[-1]), kernel_size=(ks, ks, ks), 
                              strides=(1, 1, 1), padding='same', name='out_C')(d1)
    
    #output_image = Activation("sigmoid", name='sigmoid')(output_image)

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'model_visualisation.png', show_shapes=True, show_layer_names=True)

    return model


def LSTM_Unet(img_shape, coarse_dim, ks=3, dropout=0.05, path='./'):
    """ input shape should have the form: (samples, time, rows, cols, channels)
    """
    # print message at runtime
    if(img_shape[1] == 64 and np.size(img_shape) == 4):
        print('Create 2D LSTM U-Net network with 3 levels...\n')
    elif(img_shape[1] == 128 and np.size(img_shape) == 4):
        print('Create 2D LSTM U-Net network with 4 levels...\n')
    elif(img_shape[1] == 64 and np.size(img_shape) == 5):
        print('Create 3D LSTM U-Net network with 3 levels...\n')
    elif(img_shape[1] == 128  and np.size(img_shape) == 5):
        print('Create 3D LSTM U-Net network with 4 levels...\n')
    else:
        print('???')
    
    def Conv2D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = ConvLSTM2D(filters=nr_filts, kernel_size=kernel_size, padding='same', 
                   data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = ConvLSTM2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A2' %layer_name)(a)
        return a


    def Conv3D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = ConvLSTM3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = ConvLSTM3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A2' %layer_name)(a)
        return a

    if(np.size(img_shape) == 4):
        # 2-D network
        Conv = Conv2D_Layers
        ConvLSTM = ConvLSTM2D

        def ConvTranspose(*args, **kwargs):
            # I have to isolate "name", as it needs to be passed to TimeDistributed
            if "name" in kwargs.keys():
                name = kwargs.pop("name")
            else:
                name = None
            return TimeDistributed(Conv2DTranspose(*args, **kwargs), name = name)

        def MaxPool(*args, **kwargs):
            if "name" in kwargs.keys():
                name = kwargs.pop("name")
            else:
                name = None
            return TimeDistributed(MaxPooling2D(*args, **kwargs), name = name)

    elif(np.size(img_shape) == 5):
        # 3-D network
        Conv = Conv3D_Layers
        ConvLSTM = ConvLSTM3D

        def ConvTranspose(*args, **kwargs):
            if "name" in kwargs.keys():
                name = kwargs.pop("name")
            else:
                name = None
            return TimeDistributed(Conv3DTranspose(*args, **kwargs), name = name)

        def MaxPool(*args, **kwargs):
            if "name" in kwargs.keys():
                name = kwargs.pop("name")
            else:
                name = None
            return TimeDistributed(MaxPooling3D(*args, **kwargs), name = name)

    img_input = Input(shape=img_shape, name='Image')
        
    # U-Net Encoder - upper level
    e1c = Conv(prev_layer=img_input, nr_filts=int(coarse_dim/16), kernel_size=ks, layer_name='E1')
    e1 = MaxPool(pool_size=2, name='E1_P')(e1c)
    e1 = Dropout(dropout*0.5, name='E1_D2')(e1)

    # U-Net Encoder - second level
    e2c = Conv(prev_layer=e1, nr_filts=int(coarse_dim/8), kernel_size=ks, layer_name='E2')
    e2 = MaxPool(pool_size=2, name='E2_P')(e2c)
    e2 = Dropout(dropout, name='E2_D2')(e2)

    # U-Net Encoder - third level
    e3c = Conv(prev_layer=e2, nr_filts=int(coarse_dim/4), kernel_size=ks, layer_name='E3')
    e3 = MaxPool(pool_size=2, name='E3_P')(e3c)
    e3 = Dropout(dropout, name='E3_D2')(e3) 

    if(img_shape[1] >= 64 and img_shape[1] < 128):
        # U-Net Encoder - bottom level
        b = Conv(prev_layer=e3, nr_filts=int(coarse_dim/2), kernel_size=ks, layer_name='B')
        d3 = ConvTranspose(filters=int(coarse_dim/4), kernel_size=2, strides=2, padding='same', name='D3_DC')(b)

    elif(img_shape[1] >= 128):
        # U-Net Encoder - fourth level
        e4c = Conv(prev_layer=e3, nr_filts=int(coarse_dim/2), kernel_size=ks, layer_name='E4')
        e4 = MaxPool(pool_size=2, name='E4_P')(e4c)
        e4 = Dropout(dropout, name='E4_D2')(e4)  
            
        # U-Net Encoder - bottom level
        b = Conv(prev_layer=e4, nr_filts=coarse_dim, kernel_size=ks, layer_name='B')

        # U-Net Decoder - fourth level
        d4 = ConvTranspose(filters=int(coarse_dim/2), kernel_size=ks, strides=2, padding='same', name='D4_DC')(b)
        d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
        d4 = Dropout(dropout, name='D4_D1')(d4)
        d4 = Conv(prev_layer=d4, nr_filts=int(coarse_dim/2), kernel_size=ks, layer_name='D4')

        # U-Net Decoder - third level
        d3 = ConvTranspose(filters=int(coarse_dim/4), kernel_size=ks, strides=2, padding='same', name='D3_DC')(d4)
    else:
        print('ERROR: input data have wrong dimension')

    # U-Net Decoder - third level (continue)
    d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
    d3 = Dropout(dropout, name='D3_D1')(d3)
    d3 = Conv(prev_layer=d3, nr_filts=int(coarse_dim/2), kernel_size=ks, layer_name='D3')

    # U-Net Decoder - second level
    d2 = ConvTranspose(filters=int(coarse_dim/8), kernel_size=ks, strides=2, padding='same', name='D2_DC')(d3)
    d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
    d2 = Dropout(dropout, name='D2_D1')(d2)
    d2 = Conv(prev_layer=d2, nr_filts=int(coarse_dim/4), kernel_size=ks, layer_name='D2')

    # U-Net Decoder - upper level
    d1 = ConvTranspose(filters=int(coarse_dim/16), kernel_size=ks, strides=2, padding='same', name='D1_DC')(d2)
    d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
    d1 = Dropout(dropout, name='D1_D1')(d1)
    d1 = Conv(prev_layer=d1, nr_filts=int(coarse_dim/16), kernel_size = ks, layer_name='D1')

    # Outro Layer
    output_image = ConvLSTM(
        filters=int(img_shape[-1]),
        kernel_size=ks,
        data_format='channels_last', 
        recurrent_activation='hard_sigmoid', 
        return_sequences=True,
        strides=1,
        padding='same', 
        name='out_C')(d1)
    
    output_image = Activation("sigmoid", name='sigmoid')(output_image)
    output_image = Lambda(lambda x: x[:,-1,:,:,0], name='slicer')(output_image)

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'model_visualisation_LSTM.png', show_shapes=True, show_layer_names=True)

    return model


def Unet3D_time(img_shape, coarse_dim, ks=3, dropout=0.05, path='./'):
    # print message at runtime
    depth = 3

    if(depth == 3):
        print('Create 3D U-Net network with 3 levels...\n')
    elif(depth == 4):
        print('Create 3D U-Net network with 4 levels...\n')
    else:
        print('???')

    def Conv3D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation("relu", name='relu_%s_A2' %layer_name)(a)
        return a
    
    img_input = Input(shape=img_shape, name='Image')
        
    # 3D U-Net Encoder - upper level
    e1c = Conv3D_Layers(prev_layer=img_input, nr_filts=int(coarse_dim/16),
                        kernel_size=(ks, ks, ks), layer_name='E1')
    #e1 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_P')(e1c)
    e1 = Dropout(dropout*0.5, name='E1_D2')(e1c)

    # 3D U-Net Encoder - second level
    e2c = Conv3D_Layers(prev_layer=e1, nr_filts=int(coarse_dim/8),
                        kernel_size=(ks, ks, ks), layer_name='E2')
    #e2 = MaxPooling3D(pool_size=(2, 2, 2), name='E2_P')(e2c)
    e2 = Dropout(dropout, name='E2_D2')(e2c)

    # 3D U-Net Encoder - third level
    e3c = Conv3D_Layers(prev_layer=e2, nr_filts=int(coarse_dim/4),
                        kernel_size=(ks, ks, ks), layer_name='E3')
    #e3 = MaxPooling3D(pool_size=(2, 2, 2), name='E3_P')(e3c)
    e3 = Dropout(dropout, name='E3_D2')(e3c)  

    if(depth == 3):
        # 3D U-Net Encoder - bottom level
        b = Conv3D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2), kernel_size=(ks, ks, ks), layer_name='B')
        
        d3 = Conv3DTranspose(filters=int(coarse_dim/4), kernel_size=(ks, ks, ks), 
                             strides=1, padding='same', name='D3_DC')(b)
    elif(depth == 4):
        # 3D U-Net Encoder - fourth level
        e4c = Conv3D_Layers(prev_layer=e3, nr_filts=int(coarse_dim/2),
                            kernel_size=(ks, ks, ks), layer_name='E4')
        #e4 = MaxPooling3D(pool_size=(2, 2, 2), name='E4_P')(e4c)
        e4 = Dropout(dropout, name='E4_D2')(e4c)  
            
        # 3D U-Net Encoder - bottom level
        b = Conv3D_Layers(prev_layer=e4, nr_filts=coarse_dim, kernel_size=(ks, ks, ks), layer_name='B')

        # 3D U-Net Decoder - fourth level
        d4 = Conv3DTranspose(filters=int(coarse_dim/2), kernel_size=(ks, ks, ks), 
                             strides=1, padding='same', name='D4_DC')(b)
        d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
        d4 = Dropout(dropout, name='D4_D1')(d4)
        d4 = Conv3D_Layers(prev_layer=d4, nr_filts=int(coarse_dim/2), 
                           kernel_size=(ks, ks, ks), layer_name='D4')
        
        # 3D U-Net Decoder - third level
        d3 = Conv3DTranspose(filters=int(coarse_dim/4), kernel_size=(ks, ks, ks), 
                             strides=1, padding='same', name='D3_DC')(d4)
    else:
        print('ERROR: input data have wrong dimension')

    # 3D U-Net Decoder - third level (continue)
    d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
    d3 = Dropout(dropout, name='D3_D1')(d3)
    d3 = Conv3D_Layers(prev_layer=d3, nr_filts=int(coarse_dim/2), 
                        kernel_size=(ks, ks, ks), layer_name='D3')

    # 3D U-Net Decoder - second level
    d2 = Conv3DTranspose(filters=int(coarse_dim/8), kernel_size=(ks, ks, ks), 
                        strides=1, padding='same', name='D2_DC')(d3)
    d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
    d2 = Dropout(dropout, name='D2_D1')(d2)
    d2 = Conv3D_Layers(prev_layer=d2, nr_filts=int(coarse_dim/4),
                        kernel_size=(ks, ks, ks), layer_name='D2')

    # 3D U-Net Decoder - upper level
    d1 = Conv3DTranspose(filters=int(coarse_dim/16), kernel_size=(ks, ks, ks), 
                         strides=1, padding='same', name='D1_DC')(d2)
    d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
    d1 = Dropout(dropout, name='D1_D1')(d1)
    d1 = Conv3D_Layers(prev_layer=d1, nr_filts=int(coarse_dim/16),
                       kernel_size=(ks, ks, ks), layer_name='D1')

    # Outro Layer
    output_image = Conv3D(filters=int(img_shape[-1]), kernel_size=(ks, ks, ks), 
                          strides=(1, 1, 1), padding='same', name='out_C')(d1)

    output_image = Activation("sigmoid", name='sigmoid')(output_image)

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'model_visualisation.png', show_shapes=True, show_layer_names=True)

    return model

