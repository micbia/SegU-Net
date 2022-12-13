import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation, Dropout, concatenate, Multiply
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D #, ConvLSTM3D only from tf v2.6.0
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D
from tensorflow.keras.utils import plot_model


def Unet(img_shape, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape)-1, params['depth']))

    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        #Pooling = MaxPooling2D
        Pooling = AveragePooling2D
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

    if(params['final_activation'] != None):
        output_image = Activation(params['final_activation'], name='final_activation')(output_image)
        network_layers['out_img'] = output_image

    model = Model(inputs=[img_input], outputs=[output_image], name='Unet')

    #plot_model(model, to_file=path+'UNet_visualisation.png', show_shapes=True, show_layer_names=True)
    return model


def SERENEt(img_shape1, img_shape2, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape1)-1, params['depth']))

    if(np.size(img_shape1)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        #Pooling = MaxPooling2D
        Pooling = AveragePooling2D
        ps = (2, 2)
    elif(np.size(img_shape1)-1 == 3):
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

    def BB_Layers(input_layer, i_layer):
        # pooling layer
        a = Pooling(pool_size=int(2**(i_layer)), name='Ebb%d_P' %(i_layer+1))(input_layer)
        network_layers['Ebb%d_P' %(i_layer+1)] = a
        
        # convolution block
        a = Conv_Layers(prev_layer=a, nr_filts=params['coarse_dim']//2**(params['depth']-i_layer)*img_shape1[-1], kernel_size=params['kernel_size'], layer_name='Ebb%d' %(i_layer+1))
        network_layers['Ebb%d' %(i_layer+1)] = a
        return a

    # network input layer
    img_input1 = Input(shape=img_shape1, name='Image1')
    img_input2 = Input(shape=img_shape2, name='Image2')
    network_layers = {'Image1':img_input1, 'Image2':img_input2}
    l = img_input1

    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape1[-1], kernel_size=params['kernel_size'], layer_name='E%d' %(i_l+1))
        network_layers['E%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='E%d_P' %(i_l+1))(lc)
        network_layers['E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='E%d_D' %(i_l+1))(l)
        network_layers['E%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape1[-1], kernel_size=params['kernel_size'], layer_name='B')
    network_layers['B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape1[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='E%d_DC' %(i_l+1))(l)
        network_layers['E%d_DC' %(i_l+1)] = lc

        # Bounding box block
        lbb = BB_Layers(input_layer=img_input2, i_layer=i_l)

        # Multiply 
        ml = Multiply()([network_layers['E%d_A2' %(i_l+1)], lbb])
        network_layers['M%d' %(i_l+1)] = ml

        # concatenate
        l = concatenate([lc, ml], name='concatenate_E%d_A2' %(i_l+1))
        network_layers['concatenate_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='D%d_D' %(i_l+1))(l)
        network_layers['D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape1[-1], kernel_size=params['kernel_size'], layer_name='D%d_C' %(i_l+1))
        network_layers['D%d_C' %(i_l+1)] = l

    # Outro Layer
    if(params['final_activation'] != None):
        output_image = Conv(filters=img_shape1[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_C')(l)
        network_layers['out_C'] = output_image
        output_image = Activation(params['final_activation'], name='out_img')(output_image)
    else:
        output_image = Conv(filters=img_shape1[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='out_img')(l)
    network_layers['out_img'] = output_image

    model = Model(inputs=[img_input1, img_input2], outputs=[output_image], name='Unet')

    #plot_model(model, to_file=path+'SERENEt_part2_visualisation.png', show_shapes=True, show_layer_names=True)
    return model


def FullSERENEt(img_shape, params, path='./'):
    # print message at runtime
    print('Create %dD U-Net network with %d levels...\n' %(np.size(img_shape)-1, params['depth']))

    if(np.size(img_shape)-1 == 2):
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        #Pooling = MaxPooling2D
        Pooling = AveragePooling2D
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
        a = Conv(filters=nr_filts, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
        a = BatchNormalization(name='%s_BN1' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A1' %layer_name)(a)
        
        # second block
        a = Conv2D(filters=nr_filts, kernel_size=kernel_size, padding='same', kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
        a = BatchNormalization(name='%s_BN2' %layer_name)(a)
        a = Activation(params['activation'], name='%s_A2' %layer_name)(a)
        return a

    def BB_Layers(input_layer, i_layer):
        # pooling layer
        a = Pooling(pool_size=int(2**(i_layer)), name='bb_E%d_P' %(i_layer+1))(input_layer)
        network_layers['bb_E%d_P' %(i_layer+1)] = a
        
        # convolution block
        a = Conv_Layers(prev_layer=a, nr_filts=params['coarse_dim']//2**(params['depth']-i_layer)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='bb_E%d' %(i_layer+1))
        network_layers['bb_E%d' %(i_layer+1)] = a
        return a

    # network input layer
    img_input = Input(shape=img_shape, name='Image')
    network_layers = {'Image':img_input}
    l_seg = img_input

    # ------------------------ Start SegU-Net ---------------------------
    # U-Net Encoder layers
    for i_l in range(params['depth']):
        # convolution
        lc_seg = Conv_Layers(prev_layer=l_seg, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='seg_E%d' %(i_l+1))
        network_layers['seg_E%d_A2' %(i_l+1)] = lc_seg

        # pooling
        l_seg = Pooling(pool_size=ps, name='seg_E%d_P' %(i_l+1))(lc_seg)
        network_layers['seg_E%d_P' %(i_l+1)] = l_seg

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l_seg = Dropout(do, name='seg_E%d_D' %(i_l+1))(l_seg)
        network_layers['seg_E%d_D' %(i_l+1)] = l_seg


    # bottom layer
    l = Conv_Layers(prev_layer=l_seg, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='seg_B')
    network_layers['seg_B'] = l_seg
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc_seg = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='seg_E%d_DC' %(i_l+1))(l_seg)
        network_layers['seg_E%d_DC' %(i_l+1)] = lc_seg

        # concatenate
        l_seg = concatenate([lc_seg, network_layers['seg_E%d_A2' %(i_l+1)]], name='seg_concatenate_E%d_A2' %(i_l+1))
        network_layers['seg_concatenate_E%d_A2' %(i_l+1)] = l_seg

        # dropout
        l_seg = Dropout(params['dropout'], name='seg_D%d_D' %(i_l+1))(l_seg)
        network_layers['seg_D%d_D' %(i_l+1)] = l_seg

        # convolution
        l_seg = Conv_Layers(prev_layer=l_seg, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='seg_D%d_C' %(i_l+1))
        network_layers['seg_D%d_C' %(i_l+1)] = l_seg

    # Outro Layer
    if(params['final_activation'] != None):
        output_image_seg = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='seg_out_C')(l_seg)
        network_layers['seg_out_C'] = output_image_seg
        output_image_seg = Activation(params['final_activation'], name='seg_out_img')(output_image_seg)
    else:
        output_image_seg = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='seg_out_img')(l_seg)
    network_layers['seg_out_img'] = output_image_seg

    # ------------------------ end SegU-Net ---------------------------

    # ------------------------ Start RecU-Net ---------------------------
    # U-Net Encoder layers
    l = img_input

    for i_l in range(params['depth']):
        # convolution
        lc = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='rec_E%d' %(i_l+1))
        network_layers['rec_E%d_A2' %(i_l+1)] = lc

        # pooling
        l = Pooling(pool_size=ps, name='rec_E%d_P' %(i_l+1))(lc)
        network_layers['rec_E%d_P' %(i_l+1)] = l

        # dropout
        do = 0.5*params['dropout'] if i_l == 0 else params['dropout']
        l = Dropout(do, name='rec_E%d_D' %(i_l+1))(l)
        network_layers['rec_E%d_D' %(i_l+1)] = l

    # bottom layer
    l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//img_shape[-1], kernel_size=params['kernel_size'], layer_name='rec_B')
    network_layers['rec_B'] = l
    
    # U-Net Decoder layers
    for i_l in range(params['depth'])[::-1]:
        # transposed convolution
        lc = ConvTranspose(filters=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], strides=ps, padding='same', name='rec_E%d_DC' %(i_l+1))(l)
        network_layers['rec_E%d_DC' %(i_l+1)] = lc

        # Bounding box block
        lbb = BB_Layers(input_layer=output_image_seg, i_layer=i_l)

        # Multiply 
        ml = Multiply(name='rec_M%d' %(i_l+1))([network_layers['rec_E%d_A2' %(i_l+1)], lbb])
        network_layers['rec_M%d' %(i_l+1)] = ml

        # concatenate
        l = concatenate([lc, ml], name='rec_concatenate_E%d_A2' %(i_l+1))
        network_layers['rec_concatenate_E%d_A2' %(i_l+1)] = l

        # dropout
        l = Dropout(params['dropout'], name='rec_D%d_D' %(i_l+1))(l)
        network_layers['rec_D%d_D' %(i_l+1)] = l

        # convolution
        l = Conv_Layers(prev_layer=l, nr_filts=params['coarse_dim']//2**(params['depth']-i_l)*img_shape[-1], kernel_size=params['kernel_size'], layer_name='rec_D%d_C' %(i_l+1))
        network_layers['rec_D%d_C' %(i_l+1)] = l

    # Outro Layer

    if(params['final_activation'] != None):
        output_image = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='rec_out_C')(l)
        network_layers['rec_out_C'] = output_image
        output_image = Activation(params['final_activation'], name='rec_out_img')(output_image)
    else:
        output_image = Conv(filters=img_shape[-1], kernel_size=params['kernel_size'], strides=1, padding='same', name='rec_out_img')(l)
    network_layers['rec_out_img'] = output_image

    # ------------------------ end RecU-Net ---------------------------

    model = Model(inputs=[img_input], outputs=[output_image, output_image_seg], name='Unet')

    #plot_model(model, to_file=path+'SERENEt_visualisation.png', show_shapes=True, show_layer_names=True)
    return model