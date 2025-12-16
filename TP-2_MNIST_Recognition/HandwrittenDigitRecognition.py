import tensorflow as tf
from tensorflow.python.keras import layers


def baseline_model(num_pixels, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = tf.keras.models.Sequential() # creates a empty model network

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons -> dense layer means that every neuron from the previous layer is connected to every neuron of the current layer
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # uses relu -> rectified linear unit as activation function so it can deal with non linearity

    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax')) # softmax converte raw scores into probabilities, with all the outputs summing to 1 and the highest probability being the predicted class

    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # using crossentopy because it's a multiclass classficiation, i'm using softmax in the output layer and i used one hot encoding for the labels 

    return model


def trainAndPredictMLP(x_train, y_train, x_test, y_test):

    #TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = x_train.shape[1] * x_train.shape[2]  # 28*28=784
    x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32') # shape become (num_samples, num_pixels)
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')   

    #TODO - Application 1 - Step 4 - Normalize the input values
    x_train = x_train / 255.0 # divided by 255 because of the gray scale, normalized to interval [0,1]
    x_test = x_test / 255.0

    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    y_train = tf.keras.utils.to_categorical(y_train) # one hot encoding for the labels
    y_test = tf.keras.utils.to_categorical(y_test)
    num_classes = y_test.shape[1]  # number of classes


    #TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)

    #TODO - Application 1 - Step 7 - Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2) # training the model with 10 epochs and batch size of 200

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = None  # Modify this

    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer


    #TODO - Application 2 - Step 5c - Define the pooling layer


    #TODO - Application 2 - Step 5d - Define the Dropout layer


    #TODO - Application 2 - Step 5e - Define the flatten layer


    #TODO - Application 2 - Step 5f - Define a dense layer of size 128


    #TODO - Application 2 - Step 5g - Define the output layer


    #TODO - Application 2 - Step 5h - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(x_train, y_train, x_test, y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]
    x_train = x_train.reshape((x_train.shape[0], 28, 28,1))

    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1


    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix


    #TODO - Application 2 - Step 5 - Call the CNN_model function
    model = None  # Modify this

    #TODO - Application 2 - Step 6 - Train the model


    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Tensorflow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    trainAndPredictMLP(x_train, y_train, x_test, y_test)
    

    #TODO - Application 2 - Step 1 - Train and predict on a CNN - Call the trainAndPredictCNN function


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
