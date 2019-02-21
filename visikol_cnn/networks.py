from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, GlobalAveragePooling2D, Dense, Cropping2D, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16


def unet(input_shape, n_classes):
    """Modified U-net architecture that includes padding so that the output shape equals the input shape.

    Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical
    image segmentation." International Conference on Medical image computing and computer-assisted
    intervention. Springer, Cham, 2015.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor (height, width, n_channels)
    n_classes : int
        Number of classes in the segmentation map. Note that n_classes is the depth in the 3rd dimension in the
        segmentation map.

    Returns
    -------
    model : Keras Model object

    """

    input = Input(input_shape)
    conv_block1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1')(input)
    conv_block1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(conv_block1)

    pool1 = MaxPooling2D((2, 2), name='pool1')(conv_block1)
    conv_block2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(pool1)
    conv_block2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4')(conv_block2)

    pool2 = MaxPooling2D((2, 2), name='pool2')(conv_block2)
    conv_block3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5')(pool2)
    conv_block3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6')(conv_block3)

    pool3 = MaxPooling2D((2, 2), name='pool3')(conv_block3)
    conv_block4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv7')(pool3)
    conv_block4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv8')(conv_block4)

    pool4 = MaxPooling2D((2, 2), name='pool4')(conv_block4)
    conv_block5 = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv9')(pool4)
    conv_block5 = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv10')(conv_block5)

    up_block1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='up-conv1')(conv_block5)
    up_block1 = concatenate([conv_block4, up_block1], axis=3, name='merge1')
    up_block1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv11')(up_block1)
    up_block1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv12')(up_block1)

    up_block2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='up-conv2')(up_block1)
    up_block2 = concatenate([conv_block3, up_block2], axis=3, name='merge2')
    up_block2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv13')(up_block2)
    up_block2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv14')(up_block2)

    up_block3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='up-conv3')(up_block2)
    up_block3 = concatenate([conv_block2, up_block3], axis=3, name='merge3')
    up_block3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv15')(up_block3)
    up_block3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv16')(up_block3)

    up_block4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='up-conv4')(up_block3)
    up_block4 = concatenate([conv_block1, up_block4], axis=3, name='merge4')
    up_block4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv17')(up_block4)
    up_block4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv18')(up_block4)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = Conv2D(n_classes, (1, 1), padding='same', activation=activation, name='output')(up_block4)

    model = Model(inputs=[input], output=[output])
    return model


def unet_small(input_shape, n_classes):
    """Modified U-net architecture that uses less filters and thus less parameters to train. Also, incorporates padding
        so the output shape equals the input shape.

       Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical
       image segmentation." International Conference on Medical image computing and computer-assisted
       intervention. Springer, Cham, 2015.

       Parameters
       ----------
       input_shape : tuple
           Shape of the input tensor (height, width, n_channels)
       n_classes : int
           Number of classes in the segmentation map. Note that n_classes is the depth in the 3rd dimension in the
           segmentation map.

       Returns
       -------
       model : Keras Model object

       """

    input = Input(input_shape)
    conv_block1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input)
    conv_block1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(conv_block1)

    pool1 = MaxPooling2D((2, 2), name='pool1')(conv_block1)
    conv_block2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(pool1)
    conv_block2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4')(conv_block2)

    pool2 = MaxPooling2D((2, 2), name='pool2')(conv_block2)
    conv_block3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5')(pool2)
    conv_block3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6')(conv_block3)

    pool3 = MaxPooling2D((2, 2), name='pool3')(conv_block3)
    conv_block4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7')(pool3)
    conv_block4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv8')(conv_block4)

    pool4 = MaxPooling2D((2, 2), name='pool4')(conv_block4)
    conv_block5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv9')(pool4)
    conv_block5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv10')(conv_block5)

    up_block1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='up-conv1')(conv_block5)
    up_block1 = concatenate([conv_block4, up_block1], axis=3, name='merge1')
    up_block1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv11')(up_block1)
    up_block1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv12')(up_block1)

    up_block2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='up-conv2')(up_block1)
    up_block2 = concatenate([conv_block3, up_block2], axis=3, name='merge2')
    up_block2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv13')(up_block2)
    up_block2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv14')(up_block2)

    up_block3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='up-conv3')(up_block2)
    up_block3 = concatenate([conv_block2, up_block3], axis=3, name='merge3')
    up_block3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv15')(up_block3)
    up_block3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv16')(up_block3)

    up_block4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='up-conv4')(up_block3)
    up_block4 = concatenate([conv_block1, up_block4], axis=3, name='merge4')
    up_block4 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv17')(up_block4)
    up_block4 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv18')(up_block4)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = Conv2D(n_classes, (1, 1), padding='same', activation=activation, name='output')(up_block4)

    model = Model(inputs=[input], output=[output])
    return model


def unet_original(n_classes):
    """Original U-net architecture proposed by the paper 'U-Net: Convolutional Networks for Biomedical Image Segmentation'.

    Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical
    image segmentation." International Conference on Medical image computing and computer-assisted
    intervention. Springer, Cham, 2015.

    Notice that there are no parameters for the input shape. This implementation features the original shape proposed by
    the paper with an input shape (572, 572, 1) and an output shape of (388, 388, n_classes).

    Parameters
    ----------
    n_classes : int
        Number of classes in the segmentation map. Note that n_classes is the depth in the 3rd dimension in the
        segmentation map.

    Returns
    -------
    model : Keras Model object

    """

    input = Input((572, 572, 1))
    conv_block1 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='conv1')(input)
    conv_block1 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='conv2')(conv_block1)

    pool1 = MaxPooling2D((2, 2), name='pool1')(conv_block1)
    conv_block2 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv3')(pool1)
    conv_block2 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv4')(conv_block2)

    pool2 = MaxPooling2D((2, 2), name='pool2')(conv_block2)
    conv_block3 = Conv2D(256, (3, 3), padding='valid', activation='relu', name='conv5')(pool2)
    conv_block3 = Conv2D(256, (3, 3), padding='valid', activation='relu', name='conv6')(conv_block3)

    pool3 = MaxPooling2D((2, 2), name='pool3')(conv_block3)
    conv_block4 = Conv2D(512, (3, 3), padding='valid', activation='relu', name='conv7')(pool3)
    conv_block4 = Conv2D(512, (3, 3), padding='valid', activation='relu', name='conv8')(conv_block4)

    pool4 = MaxPooling2D((2, 2), name='pool4')(conv_block4)
    conv_block5 = Conv2D(1024, (3, 3), padding='valid', activation='relu', name='conv9')(pool4)
    conv_block5 = Conv2D(1024, (3, 3), padding='valid', activation='relu', name='conv10')(conv_block5)

    up_block1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid', name='up-conv1')(conv_block5)
    crop_block4 = Cropping2D(4, name='crop_block4')(conv_block4)
    up_block1 = concatenate([crop_block4, up_block1], axis=3, name='merge1')
    up_block1 = Conv2D(512, (3, 3), padding='valid', activation='relu', name='conv11')(up_block1)
    up_block1 = Conv2D(512, (3, 3), padding='valid', activation='relu', name='conv12')(up_block1)

    up_block2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid', name='up-conv2')(up_block1)
    crop_block3 = Cropping2D(16, name='crop_block3')(conv_block3)
    up_block2 = concatenate([crop_block3, up_block2], axis=3, name='merge2')
    up_block2 = Conv2D(256, (3, 3), padding='valid', activation='relu', name='conv13')(up_block2)
    up_block2 = Conv2D(256, (3, 3), padding='valid', activation='relu', name='conv14')(up_block2)

    up_block3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid', name='up-conv3')(up_block2)
    crop_block2 = Cropping2D(40, name='crop_block2')(conv_block2)
    up_block3 = concatenate([crop_block2, up_block3], axis=3, name='merge3')
    up_block3 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv15')(up_block3)
    up_block3 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv16')(up_block3)

    up_block4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid', name='up-conv4')(up_block3)
    crop_block1 = Cropping2D(88, name='crop_block1')(conv_block1)
    up_block4 = concatenate([crop_block1, up_block4], axis=3, name='merge4')
    up_block4 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='conv17')(up_block4)
    up_block4 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='conv18')(up_block4)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = Conv2D(n_classes, (1, 1), padding='same', activation=activation, name='output')(up_block4)

    model = Model(inputs=[input], output=[output])
    return model


def inceptionv3(input_shape, n_classes, weights='imagenet'):
    """Inception archictecture for image classification. 

    Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision
    and pattern recognition. 2015.
    
    Parameters
    ----------
    input_shape : tuple
       Shape of the input tensor (height, width, n_channels)
    n_classes : int
        Number of predicted classes in the output.
    weights : str
        Indicates which weights to use for training. Options are: 'imagenet' or None.

    Returns
    -------
    model : Keras Model object

    """

    base_model = InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dense(1024, activation='relu', name='dense')(x)
    if n_classes == 1:
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        predictions = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=predictions)
    return model


def vgg16(input_shape, n_classes, weights='imagenet'):
    """VGG16 Architecture for image classification.

    Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition."
    arXiv preprint arXiv:1409.1556 (2014).

    Parameters
    ----------
    input_shape : tuple
       Shape of the input tensor (height, width, n_channels)
    n_classes : int
        Number of predicted classes in the output.
    weights : str
        Indicates which weights to use for training. Options are: 'imagenet' or None.

    Returns
    -------
    model : Keras Model object

    """

    base_model = VGG16(include_top=False, input_shape=input_shape, weights=weights)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    if n_classes == 1:
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        predictions = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=predictions)
    return model


