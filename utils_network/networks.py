import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation, Dropout, concatenate, Multiply
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D #, ConvLSTM3D only from tf v2.6.0
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.utils import plot_model


def ChonkyBoy(img_shape, params, path='./'):
    print('Combine SegU-Net and RecU-Net (%dD U-Net network with %d levels).\n' %(np.size(img_shape)-1, params['depth']))
    
    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Pooling = MaxPooling2D
        ps = (2, 2)
    elif(np.size(img_shape)-1 == 3):
        Conv = Conv3D
        Pooling = MaxPooling3D
        ConvTranspose = Conv3DTranspose
        ps = (2, 2, 2)
    else:
        print('???')

    def Conv_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = Conv(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image':img_input}
    l = img_input

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='E%d' %(i_l+1))
        network_layers['E%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='E%d_P' %(i_l+1))(lc)
        network_layers['E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='E%d_D' %(i_l+1))(l)
        network_layers['E%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='B')
    network_layers['B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='E%d_DC' %(i_l+1))(l)
        network_layers['E%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['E%d_A2' %(i_l+1)]], name='concatenate1_E%d_A2' %(i_l+1))
        network_layers['concatenate1_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='D%d_D' %(i_l+1))(l)
        network_layers['D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='D%d_C' %(i_l+1))
        network_layers['D%d_C' %(i_l+1)] = l

    # Outro Layer
    out_C = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C')(l)
    network_layers['out_C'] = out_C

    output_image_seg = Activation('sigmoid', name='out_imgSeg')(out_C)
    network_layers['out_imgSeg'] = output_image_seg
    # -----------------------------------------------------------------------------------------------------------------------------
    # network input layer
    img_input2 = Multiply(name='Image2')([output_image_seg, img_input])
    network_layers['Image2'] = img_input2
    l = img_input2

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='encoder%d' %(i_l+1))
        network_layers['encoder%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='encoder%d_P' %(i_l+1))(lc)
        network_layers['encoder%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='encoder%d_D' %(i_l+1))(l)
        network_layers['encoder%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='bottom_layer')
    network_layers['bottom_layer'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='decoder%d_DC' %(i_l+1))(l)
        network_layers['decoder%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['encoder%d_A2' %(i_l+1)]], name='concatenate2_E%d_A2' %(i_l+1))
        network_layers['concatenate2_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='decoder%d_D' %(i_l+1))(l)
        network_layers['decoder%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='decoder%d_C' %(i_l+1))
        network_layers['decoder%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image_rec = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_imgRec')(l)
    network_layers['out_imgRec'] = output_image_rec

    if(params['final_activation'] != None):
        output_image_rec = Activation(params['final_activation'], name='output_imgRec')(output_image_rec)
        network_layers['out_imgRec'] = output_image_rec

    model = Model(inputs=[img_input], outputs=[output_image_rec, output_image_seg], name='ChonkyBoy')

    plot_model(model, to_file=path+'ChonkyBoy_visualisation.png', show_shapes=True, show_layer_names=True)
    return model


def ChonkyBoy3(img_shape, params, path='./'):
    print('Combine SegU-Net and RecU-Net (%dD U-Net network with %d levels).\n' %(np.size(img_shape)-1, params['depth']))
    
    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Pooling = MaxPooling2D
        ps = (2, 2)
    elif(np.size(img_shape)-1 == 3):
        Conv = Conv3D
        Pooling = MaxPooling3D
        ConvTranspose = Conv3DTranspose
        ps = (2, 2, 2)
    else:
        print('???')

    def Conv_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = Conv(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image':img_input}
    l = img_input

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='E%d' %(i_l+1))
        network_layers['E%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='E%d_P' %(i_l+1))(lc)
        network_layers['E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='E%d_D' %(i_l+1))(l)
        network_layers['E%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='B')
    network_layers['B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='E%d_DC' %(i_l+1))(l)
        network_layers['E%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['E%d_A2' %(i_l+1)]], name='concatenate1_E%d_A2' %(i_l+1))
        network_layers['concatenate1_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='D%d_D' %(i_l+1))(l)
        network_layers['D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='D%d_C' %(i_l+1))
        network_layers['D%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image_seg = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_imgSeg')(l)
    network_layers['out_imgSeg'] = output_image_seg
    # -----------------------------------------------------------------------------------------------------------------------------
    # network input layer
    img_input2 = Multiply(name='Image2')([output_image_seg, img_input])
    network_layers['Image2'] = img_input2
    l = img_input2

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='encoder%d' %(i_l+1))
        network_layers['encoder%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='encoder%d_P' %(i_l+1))(lc)
        network_layers['encoder%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='encoder%d_D' %(i_l+1))(l)
        network_layers['encoder%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='bottom_layer')
    network_layers['bottom_layer'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='decoder%d_DC' %(i_l+1))(l)
        network_layers['decoder%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['encoder%d_A2' %(i_l+1)]], name='concatenate2_E%d_A2' %(i_l+1))
        network_layers['concatenate2_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='decoder%d_D' %(i_l+1))(l)
        network_layers['decoder%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='decoder%d_C' %(i_l+1))
        network_layers['decoder%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image_rec = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_imgRec')(l)
    network_layers['out_imgRec'] = output_image_rec

    if(params['final_activation'] != None):
        output_image_rec = Activation(params['final_activation'], name='output_imgRec')(output_image_rec)
        network_layers['out_imgRec'] = output_image_rec

    # Regression branch
    l = BatchNormalization(name='regression_BN')(network_layers['encoder4_P'])
    l = Flatten(name='regression_F')(l)
    l = Dropout(params['dropout'], name='regression_D')(l)
    
    l = Dense(640, activation='relu', name='regression_FC1')(l)
    l = BatchNormalization(name='regression_BN1')(l)

    l = Dense(512, activation='relu', name='regression_FC2')(l)
    l = BatchNormalization(name='regression_BN2')(l)

    nr_reclayer = 7
    for i_l in range(nr_reclayer):
        lname = 'out_recAstro' if(i_l==6) else 'regression_FC%d' %(i_l+3)
        l = Dense(256//2**(i_l), activation='relu', name=lname)(l)
    
    out_params = l

    model = Model(inputs=[img_input], outputs=[output_image_rec, out_params, output_image_seg], name='ChonkyBoy')

    plot_model(model, to_file=path+'ChonkyBoy_visualisation.png', show_shapes=True, show_layer_names=True)
    return model


def Unet(img_shape, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape)-1, params['depth']))

    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Pooling = MaxPooling2D
        ps = (2, 2)
    elif(np.size(img_shape)-1 == 3):
        Conv = Conv3D
        Pooling = MaxPooling3D
        ConvTranspose = Conv3DTranspose
        ps = (2, 2, 2)
    else:
        print('???')

    def Conv_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = Conv(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image':img_input}
    l = img_input

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='E%d' %(i_l+1))
        network_layers['E%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='E%d_P' %(i_l+1))(lc)
        network_layers['E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='E%d_D' %(i_l+1))(l)
        network_layers['E%d_D' %(i_l+1)] = l


    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='B')
    network_layers['B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='E%d_DC' %(i_l+1))(l)
        network_layers['E%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['E%d_A2' %(i_l+1)]], name='concatenate_E%d_A2' %(i_l+1))
        network_layers['concatenate_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='D%d_D' %(i_l+1))(l)
        network_layers['D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='D%d_C' %(i_l+1))
        network_layers['D%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C')(l)
    network_layers['out_C'] = output_image

    #output_image = Activation(params['final_activation'], name='final_activation')(output_image)
    #network_layers['out_img'] = output_image

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'UNet_visualisation.png', show_shapes=True, show_layer_names=True)
    return model



def LSTM_Unet(img_shape, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape)-1, params['depth']))

    ps = (2, 2)

    def Conv_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = TimeDistributed(Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name))(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = TimeDistributed(Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name))(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    def ConvLSTM_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = ConvLSTM2D(filters=nr_filts, kernel_size=kernel_size, padding='same', 
                       data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                       kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = ConvLSTM2D(filters=nr_filts, kernel_size=kernel_size, padding='same', 
                       data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True,
                       kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image':img_input}
    l = img_input

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = ConvLSTM_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='E%d' %(i_l+1))
        network_layers['E%d_A2' %(i_l+1)] = lc

        # pooling
        l = TimeDistributed(MaxPooling2D(pool_size=ps, name='E%d_P' %(i_l+1)))(lc)
        network_layers['E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='E%d_D' %(i_l+1))(l)
        network_layers['E%d_D' %(i_l+1)] = l


    # bottom layer
    l = ConvLSTM_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='B')
    network_layers['B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = TimeDistributed(Conv2DTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='E%d_DC' %(i_l+1)))(l)
        network_layers['E%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['E%d_A2' %(i_l+1)]], name='concatenate_E%d_A2' %(i_l+1))
        network_layers['concatenate_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='D%d_D' %(i_l+1))(l)
        network_layers['D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='D%d_C' %(i_l+1))
        network_layers['D%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image = TimeDistributed(Conv2D(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C'))(l)
    #output_image = Conv3D(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C1')(l)
    #output_image = Conv3D(filters=img_shape[-1], data_format='channels_first', kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C')(output_image)
    network_layers['out_C'] = output_image

    output_image = Activation(params['final_activation'], name='final_activation')(output_image)
    network_layers['out_img'] = output_image

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    plot_model(model, to_file=path+'LSTMUNet_visualisation.png', show_shapes=True, show_layer_names=True)
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



def Unet_Reg(img_shape, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape)-1, params['depth']))

    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Pooling = MaxPooling2D
        ps = (2, 2)
    elif(np.size(img_shape)-1 == 3):
        Conv = Conv3D
        Pooling = MaxPooling3D
        ConvTranspose = Conv3DTranspose
        ps = (2, 2, 2)
    else:
        print('???')

    def Conv_Layers(prev_layer, kernel_size, nr_filts, layer_name):
        # first block
        a = Conv(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        # second block
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same',
                   kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a
    
    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image': img_input}
    l = img_input

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='encoder%d' %(i_l+1))
        network_layers['encoder%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='encoder%d_P' %(i_l+1))(lc)
        network_layers['encoder%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='encoder%d_D' %(i_l+1))(l)
        network_layers['encoder%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='bottom_layer')
    network_layers['bottom_layer'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='decoder%d_DC' %(i_l+1))(l)
        network_layers['decoder%d_DC' %(i_l+1)] = lc

        # concatenate
        l = concatenate([lc, network_layers['encoder%d_A2' %(i_l+1)]], name='concatenate_E%d_A2' %(i_l+1))
        network_layers['concatenate_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='decoder%d_D' %(i_l+1))(l)
        network_layers['decoder%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='decoder%d_C' %(i_l+1))
        network_layers['decoder%d_C' %(i_l+1)] = l

    # Outro Layer
    output_image = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='output_img')(l)
    network_layers['output_img'] = output_image

    if(params['final_activation'] != None):
        output_image = Activation(params['final_activation'], name='output_img')(output_image)
        network_layers['out_img'] = output_image

    # Regression branch
    l = BatchNormalization(name='regression_BN')(network_layers['encoder4_P'])
    l = Flatten(name='regression_F')(l)
    l = Dropout(params['dropout'], name='regression_D')(l)
    
    l = Dense(640, activation='relu', name='regression_FC1')(l)
    l = BatchNormalization(name='regression_BN1')(l)

    l = Dense(512, activation='relu', name='regression_FC2')(l)
    l = BatchNormalization(name='regression_BN2')(l)

    nr_reclayer = 7
    for i_l in range(nr_reclayer):
        lname = 'output_rec' if(i_l==6) else 'regression_FC%d' %(i_l+3)
        l = Dense(256//2**(i_l), activation='relu', name=lname)(l)
    
    out_params = l

    model = Model(inputs=[img_input], outputs=[output_image, out_params], name='Unet_Reg')

    plot_model(model, to_file=path+'UNetReg_visualisation.png', show_shapes=True, show_layer_names=True)
    return model