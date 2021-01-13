import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')

class DeblurringCNN(tf.keras.Model):
    
    """ 
        The first version of our architecture model 
        A seven layer structure that convolves and then deconvolves to reconstruct
        the sharpened image from the input blurred image.
    """

    def __init__(self):
        super(DeblurringCNN, self).__init__()
        self.model = tf.keras.Sequential(
            [
                Conv2D(filters=128, kernel_size=17, strides=1, padding='same', 
                       activation='relu', input_shape=(128,128,3)),
                Conv2D(filters=256, kernel_size=1, strides=1, padding='same',
                       activation='relu'),
                Conv2DTranspose(filters=256, kernel_size=1, strides=1, padding='same',
                                activation='relu'),
                Conv2DTranspose(filters=128, kernel_size=7, strides=1, padding='same',
                                activation='relu'),
                Conv2DTranspose(filters=3, kernel_size=7, strides=1, padding='same',
                                activation='sigmoid'),
            ]
        )
        # Default Adam Optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        # Loss function
        self.loss = tf.keras.losses.mean_squared_error
        
        
    def compile(self):
        self.model.compile(optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=['accuracy']
        )
    
    def fit(self, x_train, y_train, 
            epochs=5, batch_size=64, validation_split=0.1):
        
        self.model.fit(x_train, y_train,
                epochs=epochs, batch_size=batch_size, 
                validation_split=validation_split, 
                verbose=1)
    
    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test,
                            verbose=1)
        return loss, accuracy
    
    def call(self, inputs):
        """
            Call the model over the inputs
        """ 
        return self.model(inputs)

    def predict(self, inputs):
        """
            Predict over the given inputs
        """
        return self.model.predict(inputs)