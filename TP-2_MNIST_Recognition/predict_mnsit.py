import tensorflow as tf
import numpy as np
from tensorflow.keras import layers   # I had to change this os it would work with the model used in the baseline_model 



def baseline_model(num_pixels, num_classes): # copied from before

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = tf.keras.models.Sequential() # creates a empty model network

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons -> dense layer means that every neuron from the previous layer is connected to every neuron of the current layer
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # uses relu -> rectified linear unit as activation function so it can deal with non linearity

    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax')) # softmax converte raw scores into probabilities, with all the outputs summing to 1 and the highest probability being the predicted class

    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # using crossentopy because it's a multiclass classficiation, i'm using softmax in the output layer and i used one hot encoding for the labels 

    return model


def main():
    (_,_), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # load the MNIST dataset

    num_pixels = x_test.shape[1] * x_test.shape[2]  # 28*28=784
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')   # reshape to (num_samples, num_pixels)

    x_test = x_test / 255.0 # normalize to [0,1]

    num_classes = 10
    model = baseline_model(num_pixels, num_classes)

    model.load_weights("mlp_mnist_model.weights.h5")  # load the trained model weights
    print("Loaded model weights from mlp_mnist_model.weights.h5")

    # first 5 samples
    x_first5 = x_test[:5]  
    y_first5 = y_test[:5]

    predictions = model.predict(x_first5)  # predict the classes for the first 5 samples
    predicted_classes = np.argmax(predictions, axis=1)  # get the class with highest probability

    for i in range(5):
        print(f"Sample {i+1}: True label: {y_first5[i]}, Predicted label: {predicted_classes[i]}")


if __name__ == "__main__":
    main()