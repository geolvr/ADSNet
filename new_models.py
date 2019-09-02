# -*- coding: utf-8 -*-
from keras.models import Model, load_model
from keras.layers import Input,ConvLSTM2D,TimeDistributed,concatenate,Add,Bidirectional,Concatenate, dot, add, multiply, \
   Activation, Reshape, Dense, RepeatVector, Dropout, Permute
from keras.layers.convolutional import Conv3D,Conv2D,SeparableConv2D,Cropping3D,Conv2DTranspose, UpSampling2D, \
   MaxPooling3D, AveragePooling3D, UpSampling3D, DepthwiseConv2D, MaxPooling2D
from keras.layers import Lambda
from keras.layers import regularizers
from generator import num_frames, fea_dim, wrf_fea_dim, num_frames_truth
from keras import backend as K
from keras import initializers as KI

def ADSNet_Plain():
    # encoder: layers definition && data flow  --------------------------------------
    # CNN module 1 -------------------------------------
    encoder_inputs = Input(shape=(num_frames_truth, 159, 159, 1), name='encoder_inputs')  # shape=(bs, 3, 159, 159, 1), "bs" means "batch size"
    encoder_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_1')(encoder_inputs)
    encoder_conv2d_1 = TimeDistributed(Activation('relu'))(encoder_conv2d_1)
    encoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_2')(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Activation('relu'))(encoder_conv2d_2)
    encoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_2)

    # ---------------------------------------------------
    _, en_h, en_c = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_sequences=True, return_state=True, padding='same',
                               name='en_convlstm')(encoder_conv2d_2)
    # --------------------------------------------------------------------------------
    # # encoder to decoder: layers definition && data flow  --------------------
    en_h = Conv2D(filters=128, kernel_size=(1, 1), padding="same", name='en_de_h', activation='relu')(en_h)
    en_c = Conv2D(filters=128, kernel_size=(1, 1), padding="same", name='en_de_c', activation='relu')(en_c)
    # --------------------------------------------------------------------------------
    # decoder: layers definition && dataflow -----------------------------------------
    decoder_inputs = Input(shape=(num_frames, 159, 159, fea_dim), name='decoder_inputs')  # (bs, 12, 159, 159, fea_dim)
    # CNN module 2 -----------------------------------------------------------
    decoder_conv2d_1 = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), padding='same'), name='de_conv2d_1')(decoder_inputs)
    decoder_conv2d_1 = TimeDistributed(Activation('relu'))(decoder_conv2d_1)
    decoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_1)
    decoder_conv2d_2 = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), padding='same'), name='de_conv2d_2')(decoder_conv2d_1)
    decoder_conv2d_2 = TimeDistributed(Activation('relu'))(decoder_conv2d_2)
    decoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_2)
    # -------------------------------------------------------------------------
    decoder_convlstm = ConvLSTM2D(filters=128, return_sequences=True,
                             kernel_size=(5, 5), name='de_convlstm_f', padding='same')([decoder_conv2d_2,en_h,en_c])
    # DCNN module ------------------------------------------------------------
    decoder_conv2dT_1 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_1')(decoder_convlstm)
    decoder_conv2dT_1 = TimeDistributed(Activation('relu'))(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_2')(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Activation('relu'))(decoder_conv2dT_2)
    decoder_outputs = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')(
        decoder_conv2dT_2)
    # ---------------------------------------------------------------------------------
    decoder_outputs = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))(decoder_outputs)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)

    return Model([decoder_inputs, encoder_inputs], decoder_outputs, name='ADSNet_Plain')

def ADSNet_W():
    # decoder: layers definition && dataflow -----------------------------------------
    inputs = Input(shape=(num_frames, 159, 159, fea_dim), name='decoder_inputs')  # (bs, 12, 159, 159, fea_dim)
    # CNN module 2 -----------------------------------------------------------
    decoder_conv2d_1 = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), padding='same'), name='de_conv2d_1')(inputs)
    decoder_conv2d_1 = TimeDistributed(Activation('relu'))(decoder_conv2d_1)
    decoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_1)
    decoder_conv2d_2 = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), padding='same'), name='de_conv2d_2')(decoder_conv2d_1)
    decoder_conv2d_2 = TimeDistributed(Activation('relu'))(decoder_conv2d_2)
    decoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_2)
    # -------------------------------------------------------------------------
    decoder_convlstm = ConvLSTM2D(filters=128, return_sequences=True,
                             kernel_size=(5, 5), name='de_convlstm_f', padding='same')(decoder_conv2d_2)
    # DCNN module ------------------------------------------------------------
    decoder_conv2dT_1 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_1')(decoder_convlstm)
    decoder_conv2dT_1 = TimeDistributed(Activation('relu'))(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_2')(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Activation('relu'))(decoder_conv2dT_2)
    decoder_outputs = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')(
        decoder_conv2dT_2)
    # ---------------------------------------------------------------------------------
    decoder_outputs = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))(decoder_outputs)
    outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)

    return Model(inputs, outputs, name='ADSNet_W')

def ADSNet_O():
    # encoder: layers definition && data flow  --------------------------------------
    # CNN module 1 -------------------------------------
    encoder_inputs = Input(shape=(num_frames_truth, 159, 159, 1), name='encoder_inputs')  # (bs, 3, 159, 159, 1)
    encoder_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_1')(encoder_inputs)
    encoder_conv2d_1 = TimeDistributed(Activation('relu'))(encoder_conv2d_1)
    encoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_2')(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Activation('relu'))(encoder_conv2d_2)
    encoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_2)

    # ---------------------------------------------------
    _, en_h, en_c = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_sequences=True, return_state=True, padding='same',
                               name='en_convlstm')(encoder_conv2d_2)
    # --------------------------------------------------------------------------------
    # # encoder to decoder: layers definition && data flow  --------------------
    en_h = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='en_de_h', activation='relu')(en_h)
    en_c = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='en_de_c', activation='relu')(en_c)
    # --------------------------------------------------------------------------------
    # decoder: layers definition && dataflow -----------------------------------------
    # CNN module 2 -----------------------------------------------------------
    de_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same',activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same',activation='relu'), name='de_conv2d_2')
    # -------------------------------------------------------------------------
    # DCNN module ------------------------------------------------------------
    de_conv2dT_1 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='relu'),
                                    name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='relu'),
                                   name='de_conv2dT_2')
    de_conv_out = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_conv_out')
    # ---------------------------------------------------------------------------------
    de_convlstm = ConvLSTM2D(filters=16, return_sequences=True, return_state=True,
                                  kernel_size=(5, 5), name='de_convlstm', padding='same')
    decoder_input_t = Cropping3D(data_format='channels_last', cropping=((num_frames_truth - 1, 0), (0, 0), (0, 0)))(encoder_inputs)
    out_list = []
    de_h = en_h
    de_c = en_c
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    sigmoid = Activation('sigmoid')
    for t in range(num_frames):
        decoder_conv2d_1 = de_conv2d_1(decoder_input_t)
        decoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_1)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(decoder_conv2d_2)
        decoder_convlstm_t, de_h, de_c = de_convlstm([decoder_conv2d_2, de_h, de_c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm_t)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_t = de_conv_out(decoder_conv2dT_2)
        decoder_out_t = cropper(decoder_out_t)
        out_list.append(decoder_out_t)
        decoder_input_t = sigmoid(decoder_out_t)

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(out_list)  # (bs, 12, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    return Model(encoder_inputs, decoder_outputs, name='ADSNet_O')

def ADSNet():
    # encoder: layers definition && data flow  --------------------------------------
    # CNN module 1 -------------------------------------
    encoder_inputs = Input(shape=(num_frames_truth, 159, 159, 1), name='encoder_inputs')  # (bs, 3, 159, 159, 1)
    encoder_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_1')(encoder_inputs)
    encoder_conv2d_1 = TimeDistributed(Activation('relu'))(encoder_conv2d_1)
    encoder_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), padding='same'),
                                       name='en_conv2d_2')(encoder_conv2d_1)
    encoder_conv2d_2 = TimeDistributed(Activation('relu'))(encoder_conv2d_2)
    encoder_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(encoder_conv2d_2)

    # ---------------------------------------------------
    _, en_h, en_c = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_sequences=True, return_state=True, padding='same',
                               name='en_convlstm')(encoder_conv2d_2)
    # --------------------------------------------------------------------------------
    # # encoder to decoder: layers definition && data flow  --------------------
    en_h = Conv2D(filters=128, kernel_size=(1, 1), padding="same", name='en_de_h', activation='relu')(en_h)
    en_c = Conv2D(filters=128, kernel_size=(1, 1), padding="same", name='en_de_c', activation='relu')(en_c)
    # --------------------------------------------------------------------------------
    # decoder: layers definition && dataflow -----------------------------------------
    decoder_inputs = Input(shape=(num_frames, 159, 159, fea_dim), name='decoder_inputs')  # (bs, 12, 159, 159, fea_dim)
    norm_inputs = Reshape((num_frames, 159 * 159, fea_dim))(decoder_inputs)
    norm_inputs = Lambda(_min_max, arguments={'axis': 2})(norm_inputs)
    norm_inputs = Reshape((num_frames, 159, 159, fea_dim))(norm_inputs)
    # CNN module 2 -----------------------------------------------------------
    de_conv2d_1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')
    de_conv2d_2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')

    # -------------------------------------------------------------------------
    # attention module ----------------------------------------------------------------
    att_conv2d_1 = TimeDistributed(DepthwiseConv2D(kernel_size=(5, 5), padding='same', depth_multiplier=1),
                                   name='att_conv2d_1')(norm_inputs)
    att_conv2d_1 = TimeDistributed(Activation('relu'))(att_conv2d_1)
    att_conv2d_1 = TimeDistributed(MaxPooling2D(padding='same'))(att_conv2d_1)
    att_conv2d_2 = TimeDistributed(DepthwiseConv2D(kernel_size=(5, 5), padding='same', depth_multiplier=1),
                                   name='att_conv2d_2')(att_conv2d_1)
    att_conv2d_2 = TimeDistributed(Activation('relu'))(att_conv2d_2)
    att_conv2d_2 = TimeDistributed(MaxPooling2D(padding='same'))(att_conv2d_2)
    att_conv_hc_to_x = Conv2D(filters=1, name='att_conv_hc_to_x', kernel_size=(1, 1))
    att_conv_x = DepthwiseConv2D(name='att_conv_x', kernel_size=(1, 1))
    de_convlstm = ConvLSTM2D(filters=128, return_sequences=True, return_state=True,
                             kernel_size=(5, 5), name='de_convlstm_f', padding='same')
    alpha_list = []
    out_list = []
    att_h = en_h
    att_c = en_c
    for t in range(num_frames):
        _att_x_t = Cropping3D(data_format='channels_last', cropping=((t, num_frames - t - 1), (0, 0), (0, 0)))(
            att_conv2d_2)  # (bs,1,40,40,fea_dim)
        norm_x_t = Cropping3D(data_format='channels_last', cropping=((t, num_frames - t - 1), (0, 0), (0, 0)))(
            norm_inputs)  # (bs,1,159,159,fea_dim)
        att_x_t = Lambda(lambda x: K.squeeze(x, axis=1))(_att_x_t)  # (bs,40,40,fea_dim)
        att_x_t = att_conv_x(att_x_t)  # (bs,40,40,fea_dim)
        hc = Concatenate(axis=-1)([att_h, att_c])  # output: (bs,40,40,128*2)
        hc = att_conv_hc_to_x(hc)  # (bs,40,40,1)
        hc_x = multiply([hc, att_x_t])  # (bs,40,40,fea_dim)
        e = Lambda(lambda x: K.sum(x, axis=[1, 2], keepdims=False))(hc_x)  # (bs,fea_dim)
        e = Lambda(lambda x: x * 0.025)(e)  # (bs,fea_dim)
        alpha = Activation('softmax')(e)  # output: (bs,fea_dim)
        norm_x_t = multiply([alpha, norm_x_t])  # output: (bs,1,40,40,fea_dim)
        norm_x_t = TimeDistributed(de_conv2d_1)(norm_x_t)
        norm_x_t = TimeDistributed(MaxPooling2D(padding='same'))(norm_x_t)
        norm_x_t = TimeDistributed(de_conv2d_2)(norm_x_t)
        norm_x_t = TimeDistributed(MaxPooling2D(padding='same'))(norm_x_t)  # (bs, 1, 40, 40, 32)
        att_o, att_h, att_c = de_convlstm([norm_x_t, att_h, att_c])
        alpha_list.append(alpha)
        out_list.append(att_o)
    decoder_convlstm = Concatenate(axis=1)(out_list)  # output: (bs,12,40,40,128)
    # ---------------------------------------------------------------------------------
    # DCNN module ------------------------------------------------------------
    decoder_conv2dT_1 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_1')(decoder_convlstm)
    decoder_conv2dT_1 = TimeDistributed(Activation('relu'))(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'),
                                        name='de_conv2dT_2')(decoder_conv2dT_1)
    decoder_conv2dT_2 = TimeDistributed(Activation('relu'))(decoder_conv2dT_2)
    decoder_outputs = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')(
        decoder_conv2dT_2)
    # ---------------------------------------------------------------------------------
    decoder_outputs = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))(decoder_outputs)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)

    return Model([decoder_inputs, encoder_inputs], decoder_outputs, name='ADSNet')

def StepDeep_model():
    WRF_inputs = Input(shape=(num_frames, 159, 159, fea_dim))   # (bs, 12, 159, 159, fea_dim)
    _history_inputs = Input(shape=(num_frames_truth, 159, 159, 1))  # (bs,3,159,159,1)
    history_inputs = Permute((4, 2, 3, 1))(_history_inputs)              #  (bs, 1, 159, 159, 3)
    conv_1 = Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same', name='conv3d_1')(WRF_inputs)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_2')(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_3 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', name='conv3d_3')(conv_2)
    conv_3 = Activation('relu')(conv_3)
    conv_4 = Conv3D(filters=128, kernel_size=(5, 1, 1), padding='same', name='conv3d_4')(conv_3)
    conv_4 = Activation('relu')(conv_4)
    conv_5 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_5')(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name='conv3d_6')(conv_5)
    conv_6 = Activation('relu')(conv_6)
    steps = []
    for i in range(num_frames):
        conv_6_i = Cropping3D(cropping=((i, num_frames - i - 1), (0, 0), (0, 0)))(conv_6)   # (bs, 1, 159, 159, fea_dim)
        conv2d_in = concatenate([history_inputs, conv_6_i], axis=-1)                        # (bs, 1, 159, 159, fea_dim+3)
        conv2d_in = Lambda(lambda x: K.squeeze(x, axis=1))(conv2d_in)  # (bs, 159, 159, fea_dim+3)
        conv2d_1_i = Conv2D(filters=64, kernel_size=(7, 7), padding='same', name='conv2d_1_%d' % i)(conv2d_in)
        conv2d_1_i = Activation('relu')(conv2d_1_i)
        conv2d_2_i = Conv2D(filters=1, kernel_size=(7, 7), padding='same', name='conv2d_2_%d' % i)(conv2d_1_i)
        steps.append(conv2d_2_i)
    conv_out = concatenate(steps, axis=1)  # (bs, 12, 159, 159, 1)
    outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(conv_out)
    return Model([WRF_inputs, _history_inputs], outputs, name='StepDeep_model')
