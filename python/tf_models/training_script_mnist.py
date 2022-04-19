from dlranet import VariationalAutoEncoder, ReferenceNet, PartDLRANet

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def main3():
    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9
    model = PartDLRANet(input_dim=input_dim, output_dim=output_dim,low_rank=10,tol=0.4)
    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()

    # specify training
    epochs = 100
    batch_size = 64
    # Build dataset
    # Prepare the training dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    (x_val, y_val) = normalize_img(x_val, y_val)

    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    (x_train, y_train) = normalize_img(x_train, y_train)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # Iterate over epochs. (Training loop)
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.

        for step, batch_train in enumerate(train_dataset):
            # 1.a) K and L Step Preproccessing
            model.dlraBlock.k_step_preprocessing()
            model.dlraBlock.l_step_preprocessing()
            # 1.b) Tape Gradients for K-Step
            model.toggle_non_s_step_training()
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=0, training=True)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_k_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_k_step, model.trainable_weights)
            grads_k_step[1] = grads_k_step[1] * 0  # Set bias grads to 0

            # 1.b) Tape Gradients for L-Step
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=1, training=True)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_l_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_l_step, model.trainable_weights)
            grads_l_step[1] = grads_l_step[1] * 0  # Set bias grads to 0

            # Gradient update for K and L
            optimizer.apply_gradients(zip(grads_k_step, model.trainable_weights))
            optimizer.apply_gradients(zip(grads_l_step, model.trainable_weights))

            # Postprocessing K and L
            model.dlraBlock.k_step_postprocessing_adapt()
            model.dlraBlock.l_step_postprocessing_adapt()

            # S-Step Preprocessing
            model.dlraBlock.s_step_preprocessing()
            model.toggle_s_step_training()

            # 3.b) Tape Gradients
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=2, training=True)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            # 3.c) Apply Gradients
            grads_s = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_s, model.trainable_weights)
            optimizer.apply_gradients(zip(grads_s, model.trainable_weights))  # All gradients except K and L matrix

            # Rank Adaptivity
            model.dlraBlock.rank_adaption()

            # Network monotoring and verbosity
            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss S-Step = %.4f" % (step, loss_metric.result()))
                print("Current Rank: " + str(int(model.dlraBlock.low_rank)))

    test = model(val_dataset[0], step=0)
    # plt.plot(test_x, test.numpy(), '-.')
    # plt.plot(test_x, test_y, '--')
    # plt.show()
    return 0


def main2():
    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9
    model = ReferenceNet(output_dim=output_dim)  # input_dim gets automatically assigned
    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()

    # specify training
    epochs = 10
    batch_size = 64
    # Build dataset
    # Prepare the training dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    (x_val, y_val) = normalize_img(x_val, y_val)

    t = x_val[0]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    (x_train, y_train) = normalize_img(x_train, y_train)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out = model(batch_train[0])
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    test = model(x_val)
    # plt.plot(test_x, test.numpy(), '-.')
    # plt.plot(test_x, test_y, '--')
    # plt.show()
    return 0


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    # main()
    main3()