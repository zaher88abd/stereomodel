from model.layers import *


def visual_encoder(input):
    x = Conv2D(filters=32, kernel_size=1, strides=1, padding="valid")(input)
    dense = dens_block(x)
    concat_1 = concatenate([x, dense], axis=-1)
    x = bottleneck(concat_1, n_filters=22)
    x = down_sampling(x, n_filters=22)

    dense = dens_block(x)
    concat_2 = concatenate([x, dense], axis=-1)
    x = bottleneck(concat_2, n_filters=17)
    x = down_sampling(x, n_filters=17)

    dense = dens_block(x)
    concat_3 = concatenate([x, dense], axis=-1)
    x = bottleneck(concat_3, n_filters=14)
    x = down_sampling(x, n_filters=14)
    visual_code = dens_block(x)
    return visual_code, concat_3, concat_2, concat_1


def visual_decoder(l_encoder, l_l1, l_l2, l_l3, r_encoder, r_l1, r_l2, r_l3):
    deep_concatenate = concatenate([l_encoder, r_encoder], axis=-1)
    l = dens_block(deep_concatenate)
    l = up_sample(l, n_filters=12)
    l = concatenate([l, l_l1, r_l1])
    l = dens_block(l)

    l = up_sample(l, n_filters=12)
    l = concatenate([l, l_l2, r_l2])
    l = dens_block(l)

    l = up_sample(l, n_filters=12)
    l = concatenate([l, l_l3, r_l3])
    l = dens_block(l)
    return l


def SoftmaxLayer(inputs, n_classes, name):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(filters=32, kernel_size=1, padding='same',
               kernel_initializer='he_uniform')(inputs)
    l = Conv2D(n_classes, kernel_size=1, padding='same',
               kernel_initializer='he_uniform')(l)
    #    l = Reshape((-1, n_classes))(l)
    l = Activation('softmax', name=name)(l)  # or softmax for multi-class
    return l


def build_model(input_shape, n_classes=1):
    right_input = Input(input_shape, name="right_input")
    left_input = Input(input_shape, name="left_input")
    r_encoder, r_l1, r_l2, r_l3= visual_encoder(right_input)
    l_encoder, l_l1, l_l2, l_l3 = visual_encoder(left_input)
    visual_decoder_output = visual_decoder(l_encoder, l_l1, l_l2, l_l3, r_encoder, r_l1, r_l2, r_l3)
    output_layer = SoftmaxLayer(visual_decoder_output, n_classes, name="stem_output")
    model = Model(inputs=[left_input,right_input], outputs=output_layer)
    return model
