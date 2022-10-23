import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Conv2D, BatchNormalization, Activation, Reshape
from keras.losses import Huber
from keras.metrics import BinaryAccuracy
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from efficient_net import pretrained_efficientnet_b0


class Model(keras.models.Model):
    """

    """

    IMAGE_SIZE = 224

    def __init__(self):
        self.anchors_count = 5

        # Load the EfficientNet-B0 model
        efficientnet_b0 = pretrained_efficientnet_b0(
            include_top=False)
        efficientnet_b0.trainable = False

        inputs = Input(shape=(Model.IMAGE_SIZE, Model.IMAGE_SIZE, 3))
        outputs = efficientnet_b0(inputs)
        x = outputs[2]

        classes = Conv2D(256, 3, padding='same')(x)
        classes = BatchNormalization()(classes)
        classes = Activation('relu')(classes)

        classes = Conv2D(256, 3, padding='same')(classes)
        classes = BatchNormalization()(classes)
        classes = Activation('relu')(classes)

        classes = Conv2D(256, 3, padding='same')(classes)
        classes = BatchNormalization()(classes)
        classes = Activation('relu')(classes)

        classes = Conv2D(
            10*self.anchors_count, 3, padding='same', activation='sigmoid')(classes)
        classes = Reshape(
            (14*14*self.anchors_count, 10), name="classes_output")(classes)

        bboxes = Conv2D(256, 3, padding='same')(x)
        bboxes = BatchNormalization()(bboxes)
        bboxes = Activation('relu')(bboxes)

        bboxes = Conv2D(256, 3, padding='same')(bboxes)
        bboxes = BatchNormalization()(bboxes)
        bboxes = Activation('relu')(bboxes)

        bboxes = Conv2D(256, 3, padding='same')(bboxes)
        bboxes = BatchNormalization()(bboxes)
        bboxes = Activation('relu')(bboxes)

        bboxes = Conv2D(4*self.anchors_count, 3, padding='same')(bboxes)
        bboxes = Reshape(
            (14*14*self.anchors_count, 4), name="bboxes_output")(bboxes)

        super().__init__(inputs=[inputs], outputs=[classes, bboxes])
