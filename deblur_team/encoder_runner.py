
import matplotlib.pyplot as plt
import tensorflow as tf

from .autoencoder import AutoEncoder
from .blur_autoencoder import BlurAutoEncoder
from .var_autoencoder import VarAutoEncoder
from .image_grabber import grab_images

def train(model, optimizer, blurred_images, original_images):
    # make the blurred images into a tensor of shape (batch_size, 128, 128, 3)
    blurred_images = tf.reshape(blurred_images, [-1, 128, 128, 3])
    # cast images to float
    blurred_images = tf.cast(blurred_images, dtype=tf.float32)
    # divide by 255
    blurred_images = blurred_images / 255

    original_images = tf.reshape(original_images, [-1, 128, 128, 3])
    original_images = tf.cast(original_images, dtype=tf.float32)
    original_images = original_images / 255
    with tf.GradientTape() as tape:
        decoded = model(blurred_images)
        loss = model.loss_function(decoded, original_images)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def total_loss(model, blurred_images):
    sum_loss = 0
    blurred_images = tf.reshape(blurred_images, [-1, 128, 128, 3])
    blurred_images = tf.cast(blurred_images, dtype=tf.float32)
    blurred_images = blurred_images / 255
    predictions = model(blurred_images)
    sum_loss += model.loss_function(predictions, blurred_images).numpy()
    return sum_loss

def trainVAE(model, optimizer, blurred_images, original_images):
    # make the blurred images into a tensor of shape (batch_size, 128, 128, 3)
    blurred_images = tf.reshape(blurred_images, [-1, 128, 128, 3])
    # cast images to float
    blurred_images = tf.cast(blurred_images, dtype=tf.float32)
    # divide by 255
    blurred_images = blurred_images / 255

    original_images = tf.reshape(original_images, [-1, 128, 128, 3])
    original_images = tf.cast(original_images, dtype=tf.float32)
    original_images = original_images / 255
    with tf.GradientTape() as tape:
        decoded, encodings, mean, logvar = model(blurred_images)
        loss = model.loss_function(decoded, encodings, original_images, mean, logvar)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def total_SSD_VAE(model, blurred_images):
    sum_loss = 0
    blurred_images = tf.reshape(blurred_images, [-1, 128, 128, 3])
    blurred_images = tf.cast(blurred_images, dtype=tf.float32)
    blurred_images = blurred_images / 255
    predictions, embeddings, mean, logvar = model(blurred_images)
    sum_loss += AutoEncoder.loss_function(model, predictions, blurred_images).numpy()
    return sum_loss

"""Visualize Results"""
def showImages(model, blurred_images):
    blurred_images = tf.reshape(blurred_images, [-1, 128, 128, 3])
    blurred_images = tf.cast(blurred_images, dtype=tf.float32)
    blurred_images = blurred_images / 255
    recon = model(blurred_images)

    # pick 5 random images, and display 5 blurred on top, and 5 recon on bottom
    indices = tf.range(start=0, limit=tf.shape(blurred_images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_blurred = tf.gather(blurred_images, shuffled_indices)
    shuffled_recon = tf.gather(recon, shuffled_indices)
    fig_images = tf.concat([shuffled_recon[:5], shuffled_blurred[:5]], axis=0)
    print(shuffled_recon[0])
    cols = 5
    rows = 2
    fig = plt.figure(figsize=(20, 12))

    for i in range(1, 11):
        img = fig_images[i - 1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)

    plt.show()

encoder_dict = {"auto":AutoEncoder(), "blur":BlurAutoEncoder(), "var":VarAutoEncoder()}


def run_encoder(encoder_type="auto", n_epochs=10, batch_size=100):

    if encoder_type not in encoder_dict:
        print("Invalid encoder argument specified. Valid arguments are: auto, blur, var")
    model = encoder_dict[encoder_type]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # get images for testing and training
    train_images_blur, test_images_blur = grab_images('dataset_blurred/architecure')
    train_images, test_images = grab_images('dataset/architecure')

    # for each epoch
    for i in range(n_epochs):
        # for each batch
        for j in range(0, len(train_images), batch_size):
            print("Batch #{} of {}".format(j // batch_size + 1, len(train_images) // batch_size + 1))
            # pass in the model, optimizer, blurred images, and truth images to train
            if encoder_type == 'var':
                trainVAE(model, optimizer, train_images_blur[j:j+batch_size], train_images[j:j+batch_size])
            else:
                train(model, optimizer, train_images_blur[j:j + batch_size], train_images[j:j + batch_size])

        print("Epoch: ", i)
        if encoder_type == 'var':
            sum_loss = total_SSD_VAE(model, test_images)
        else:
            sum_loss = total_loss(model, test_images)
        print("Total Loss: {0}".format(sum_loss))
        showImages(model, test_images_blur)
