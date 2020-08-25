import numpy as np, time

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dropout, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.utils import plot_model


def Unet(img_shape, params, path='./'):
    
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
        a = Activation(params['activation'], name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='relu_%s_A2' %layer_name)(a)
        return a


    def Conv3D_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='relu_%s_A1' %layer_name)(a)
        # second layer
        a = Conv3D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='relu_%s_A2' %layer_name)(a)
        return a

    
    img_input = Input(shape=img_shape, name='Image')
        
    # U-Net Encoder - upper level
    if(np.size(img_shape) == 3):
        # 2-D network
        e1c = Conv2D_Layers(prev_layer=img_input, nr_filts=int(params['coarse_dim']/16),
                            kernel_size=params['kernel_size'], layer_name='E1')
        e1 = MaxPooling2D(pool_size=(2, 2), name='E1_P')(e1c)
        e1 = Dropout(params['dropout']*0.5, name='E1_D2')(e1)
    elif(np.size(img_shape) == 4):
        # 3-D network
        e1c = Conv3D_Layers(prev_layer=img_input, nr_filts=int(params['coarse_dim']/16),
                            kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='E1')
        e1 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_P')(e1c)
        e1 = Dropout(params['dropout']*0.5, name='E1_D2')(e1)

    # U-Net Encoder - second level
    if(np.size(img_shape) == 3):
        # 2-D network
        e2c = Conv2D_Layers(prev_layer=e1, nr_filts=int(params['coarse_dim']/8),
                            kernel_size=params['kernel_size'], layer_name='E2')
        e2 = MaxPooling2D(pool_size=(2, 2), name='E2_P')(e2c)
        e2 = Dropout(params['dropout'], name='E2_D2')(e2)
    elif(np.size(img_shape) == 4):
        # 3-D network
        e2c = Conv3D_Layers(prev_layer=e1, nr_filts=int(params['coarse_dim']/8),
                            kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='E2')
        e2 = MaxPooling3D(pool_size=(2, 2, 2), name='E2_P')(e2c)
        e2 = Dropout(params['dropout'], name='E2_D2')(e2)

    # U-Net Encoder - third level
    if(np.size(img_shape) == 3):
        # 2-D network
        e3c = Conv2D_Layers(prev_layer=e2, nr_filts=int(params['coarse_dim']/4),
                            kernel_size=params['kernel_size'], layer_name='E3')
        e3 = MaxPooling2D(pool_size=(2, 2), name='E3_P')(e3c)
        e3 = Dropout(params['dropout'], name='E3_D2')(e3) 
    elif(np.size(img_shape) == 4):
        # 3-D network
        e3c = Conv3D_Layers(prev_layer=e2, nr_filts=int(params['coarse_dim']/4),
                            kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='E3')
        e3 = MaxPooling3D(pool_size=(2, 2, 2), name='E3_P')(e3c)
        e3 = Dropout(params['dropout'], name='E3_D2')(e3)  

    if(img_shape[0] >= 64 and img_shape[0] < 128):
        # U-Net Encoder - bottom level
        if(np.size(img_shape) == 3):
            # 2-D network
            b = Conv2D_Layers(prev_layer=e3, nr_filts=int(params['coarse_dim']/2), kernel_size=(params['kernel_size'], params['kernel_size']), layer_name='B')
            
            d3 = Conv2DTranspose(filters=int(params['coarse_dim']/4), kernel_size=(params['kernel_size'], params['kernel_size']), 
                                strides=(2, 2), padding='same', name='D3_DC')(b)
        elif(np.size(img_shape) == 4):
            # 3-D network
            b = Conv3D_Layers(prev_layer=e3, nr_filts=int(params['coarse_dim']/2), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='B')
            
            d3 = Conv3DTranspose(filters=int(params['coarse_dim']/4), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                                strides=(2, 2, 2), padding='same', name='D3_DC')(b)
    elif(img_shape[0] >= 128):
        if(np.size(img_shape) == 3):
            # 2-D network
            # U-Net Encoder - fourth level
            e4c = Conv2D_Layers(prev_layer=e3, nr_filts=int(params['coarse_dim']/2),
                                kernel_size=params['kernel_size'], layer_name='E4')
            e4 = MaxPooling2D(pool_size=(2, 2), name='E4_P')(e4c)
            e4 = Dropout(params['dropout'], name='E4_D2')(e4)  
                
            # U-Net Encoder - bottom level
            b = Conv2D_Layers(prev_layer=e4, nr_filts=params['coarse_dim'], kernel_size=params['kernel_size'], layer_name='B')

            # U-Net Decoder - fourth level
            d4 = Conv2DTranspose(filters=int(params['coarse_dim']/2), kernel_size=params['kernel_size'], 
                                strides=(2, 2), padding='same', name='D4_DC')(b)
            d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
            d4 = Dropout(params['dropout'], name='D4_D1')(d4)
            d4 = Conv2D_Layers(prev_layer=d4, nr_filts=int(params['coarse_dim']/2), 
                            kernel_size=(params['kernel_size'], params['kernel_size']), layer_name='D4')

            # U-Net Decoder - third level
            d3 = Conv2DTranspose(filters=int(params['coarse_dim']/4), kernel_size=params['kernel_size'], 
                                strides=(2, 2), padding='same', name='D3_DC')(d4)
        elif(np.size(img_shape) == 4):
            # 3-D network
            # U-Net Encoder - fourth level
            e4c = Conv3D_Layers(prev_layer=e3, nr_filts=int(params['coarse_dim']/2),
                                kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='E4')
            e4 = MaxPooling3D(pool_size=(2, 2, 2), name='E4_P')(e4c)
            e4 = Dropout(params['dropout'], name='E4_D2')(e4)  
                
            # U-Net Encoder - bottom level
            b = Conv3D_Layers(prev_layer=e4, nr_filts=params['coarse_dim'], kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='B')

            # U-Net Decoder - fourth level
            d4 = Conv3DTranspose(filters=int(params['coarse_dim']/2), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                                strides=(2, 2, 2), padding='same', name='D4_DC')(b)
            d4 = concatenate([d4, e4c], name='merge_layer_E4_A2')
            d4 = Dropout(params['dropout'], name='D4_D1')(d4)
            d4 = Conv3D_Layers(prev_layer=d4, nr_filts=int(params['coarse_dim']/2), 
                            kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='D4')
            
            # U-Net Decoder - third level
            d3 = Conv3DTranspose(filters=int(params['coarse_dim']/4), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                            strides=(2, 2, 2), padding='same', name='D3_DC')(d4)
    else:
        print('ERROR: input data have wrong dimension')

    # U-Net Decoder - third level (continue)
    if(np.size(img_shape) == 3):
        # 2-D network
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(params['dropout'], name='D3_D1')(d3)
        d3 = Conv2D_Layers(prev_layer=d3, nr_filts=int(params['coarse_dim']/2), 
                           kernel_size=(params['kernel_size'], params['kernel_size']), layer_name='D3')
    elif(np.size(img_shape) == 4):
        # 3-D network
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(params['dropout'], name='D3_D1')(d3)
        d3 = Conv3D_Layers(prev_layer=d3, nr_filts=int(params['coarse_dim']/2), 
                           kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='D3')

    # U-Net Decoder - second level
    if(np.size(img_shape) == 3):
        # 2-D network
        d2 = Conv2DTranspose(filters=int(params['coarse_dim']/8), kernel_size=params['kernel_size'], 
                        strides=(2, 2), padding='same', name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
        d2 = Dropout(params['dropout'], name='D2_D1')(d2)
        d2 = Conv2D_Layers(prev_layer=d2, nr_filts=int(params['coarse_dim']/4),
                    kernel_size=(params['kernel_size'], params['kernel_size']), layer_name='D2')
    elif(np.size(img_shape) == 4):
        # 3-D network
        d2 = Conv3DTranspose(filters=int(params['coarse_dim']/8), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                            strides=(2, 2, 2), padding='same', name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
        d2 = Dropout(params['dropout'], name='D2_D1')(d2)
        d2 = Conv3D_Layers(prev_layer=d2, nr_filts=int(params['coarse_dim']/4),
                           kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='D2')

    # U-Net Decoder - upper level
    if(np.size(img_shape) == 3):
        d1 = Conv2DTranspose(filters=int(params['coarse_dim']/16), kernel_size=params['kernel_size'], 
                            strides=(2, 2), padding='same', name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
        d1 = Dropout(params['dropout'], name='D1_D1')(d1)
        d1 = Conv2D_Layers(prev_layer=d1, nr_filts=int(params['coarse_dim']/16),
                        kernel_size=(params['kernel_size'], params['kernel_size']), layer_name='D1')
    elif(np.size(img_shape) == 4):
        d1 = Conv3DTranspose(filters=int(params['coarse_dim']/16), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                            strides=(2, 2, 2), padding='same', name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
        d1 = Dropout(params['dropout'], name='D1_D1')(d1)
        d1 = Conv3D_Layers(prev_layer=d1, nr_filts=int(params['coarse_dim']/16),
                        kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), layer_name='D1')

    # Outro Layer
    if(np.size(img_shape) == 3):
        output_image = Conv2D(filters=int(img_shape[-1]), kernel_size=params['kernel_size'], 
                              strides=(1, 1), padding='same', name='out_C')(d1)
    elif(np.size(img_shape) == 4):
        output_image = Conv3D(filters=int(img_shape[-1]), kernel_size=(params['kernel_size'], params['kernel_size'], params['kernel_size']), 
                              strides=(1, 1, 1), padding='same', name='out_C')(d1)
    
    output_image = Activation("sigmoid", name='sigmoid')(output_image)

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'model_visualization.png', show_shapes=True, show_layer_names=True)

    return model