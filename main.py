#!/usr/bin/env python3
from svhn_dataset import SVHN
import efficient_net
import bboxes_utils
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import argparse
import cv2
import datetime
import os
import re

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")



def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%height%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    svhn = SVHN()
    train = svhn.train.map(lambda data: (
        data['image'], data['bboxes'], data['classes'])).take(-1)
    train = list(train.as_numpy_iterator())[:3000]

    dev = svhn.dev.map(lambda data: (data['image'])).take(-1)
    dev = list(dev.as_numpy_iterator())[:500]

    test = svhn.test.map(lambda data: (data['image'])).take(-1)
    test = list(test.as_numpy_iterator())[:500]

    my_anchors = []

    a1 = [0.15, 0.2]
    a2 = [0.2, 0.3]
    a3 = [0.3, 0.6]
    a4 = [0.4, 0.75]
    a5 = [0.5, 0.85]
    anchors_count = 5
    for y in np.linspace(0, 1, 14):
        for x in np.linspace(0, 1, 14):
            my_anchors.append((y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2))
            my_anchors.append((y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2))
            my_anchors.append((y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2))
            my_anchors.append((y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2))
            my_anchors.append((y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2))

    my_anchors = np.array(my_anchors)

    image_size = 224

    X_train = []

    all_cat_classes = []
    all_bboxes = []
    all_sample_weights = []

    X_train_multiples = []

    for image, g_bboxes, classes in train:
        height, width, _ = image.shape
        multiple_h = image_size / height
        multiple_w = image_size / width
        X_train_multiples.append([multiple_h, multiple_w])
        img = cv2.resize(image, (image_size, image_size))
        X_train.append(img)

        g_bboxes = np.array(g_bboxes)
        g_bboxes[:, 0] *= multiple_h / image_size
        g_bboxes[:, 2] *= multiple_h / image_size
        g_bboxes[:, 1] *= multiple_w / image_size
        g_bboxes[:, 3] *= multiple_w / image_size

        classes, bboxes = bboxes_utils.bboxes_training(
            my_anchors, classes, g_bboxes, 0.5)

        cat_classes = keras.utils.to_categorical(classes, 11)

        all_bboxes.append(bboxes)
        all_cat_classes.append(cat_classes)
        all_sample_weights.append(
            (cat_classes.argmax(axis=1) > 0).astype(np.float32))

    X_train = np.array(X_train)
    all_cat_classes = np.array(all_cat_classes)
    all_bboxes = np.array(all_bboxes)
    all_sample_weights = np.array(all_sample_weights)

    X_dev = []
    X_dev_multiples = []
    for image in dev:
        height, width, _ = image.shape
        multiple_h = image_size / height
        multiple_w = image_size / width
        X_dev.append(cv2.resize(image, (image_size, image_size)))
        X_dev_multiples.append([multiple_h, multiple_w])
    X_dev = np.array(X_dev)

    X_test = []
    X_test_multiples = []
    for image in test:
        height, width, _ = image.shape
        multiple_h = image_size / height
        multiple_w = image_size / width
        X_test.append(cv2.resize(image, (image_size, image_size)))
        X_test_multiples.append([multiple_h, multiple_w])
    X_test = np.array(X_test)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)
    efficientnet_b0.trainable = False

    # TODO: Create the model and train it
    input_l = keras.layers.Input(shape=(image_size, image_size, 3))
    o0, o1, o2, o3, o4, o5, *_ = efficientnet_b0(input_l)
    x = o2

    classes = keras.layers.Conv2D(256, 3, padding='same')(x)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(256, 3, padding='same')(classes)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(256, 3, padding='same')(classes)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(
        10*anchors_count, 3, padding='same', activation='sigmoid')(classes)
    classes = keras.layers.Reshape(
        (14*14*anchors_count, 10), name="classes_output")(classes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(x)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(bboxes)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(bboxes)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(4*anchors_count, 3, padding='same')(bboxes)
    bboxes = keras.layers.Reshape(
        (14*14*anchors_count, 4), name="bboxes_output")(bboxes)

    model = keras.models.Model(inputs=[input_l], outputs=[classes, bboxes])
    model.summary()

    losses = {
        'classes_output':   tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
        'bboxes_output':  tf.keras.losses.Huber(),
    }

    metrics = {
        'classes_output': keras.metrics.BinaryAccuracy(),
    }

    batch_size = 32
    epochs = 20
    decay_steps = epochs * len(train)*2 / batch_size
    lr_decayed_fn = keras.experimental.CosineDecay(
        0.01, decay_steps, alpha=0.00001)
    model.compile(optimizer=keras.optimizers.Adam(lr_decayed_fn),
                  loss=losses,
                  metrics=metrics,
                  run_eagerly=False)

    model.fit(X_train,  {'classes_output': all_cat_classes[:, :, 1:], 'bboxes_output': all_bboxes}, batch_size=batch_size, epochs=epochs, sample_weight={
              'bboxes_output': all_sample_weights})

    model.save('model.h5')
    model = keras.models.load_model('model.h5')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
