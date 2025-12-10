import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

dict_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def most_frequent(list):
    list = [x for x in list]
    return [max(set(list), key = list.count)]

# TODO - Application 1 - Step 3 - Compute the difference between the current image (img) taken from test dataset
#  with all the images from the train dataset. Return the label for the training image for which is obtained
#  the lowest score
def predictLabelNN(x_train_flatten, y_train, img):

    predictedLabel = -1
    scoreMin = float('inf')  #initialize with infinity so any computed score will be lower

    # TODO - Application 1 - Step 3a - for each image in the training list
    for idx, imgT in enumerate(x_train_flatten):

        # TODO - Application 1 - Step 3b - compute the absolute difference between img and imgT
        difference = np.abs(img - imgT) #this will get how different the two images are, the smaller the difference the more similar they are
        distance = np.sqrt(np.square(img) + np.square(imgT))  #Euclidean distance formula

        # TODO - Application 1 - Step 3c - add all pixels differences to a single number (score)
        #score = np.sum(difference) 
        score = np.sum(distance) # score for euclidean distance

        # TODO - Application 1 - Step 3d - retain the label where the minimum score is obtained
        if score < scoreMin:
            scoreMin = score
            predictedLabel = y_train[idx][0]



        #pass   #REMOVE THIS

    return predictedLabel
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 2 - Step 1 - Create a function (predictLabelKNN) that predict the label for a test image based
#  on the dominant class obtained when comparing the current image with the k-NearestNeighbor images from train
def predictLabelKNN(x_train_flatten, y_train, img):

    predictedLabel = -1
    predictions = []  # list to save the scores and associated labels as pairs  (score, label)


    #TODO - Application 2 - Step 1a - for each image in the training list
    for idx, imgT in enumerate(x_train_flatten):

        # TODO - Application 2 - Step 1b - compute the absolute difference between img and imgT
        difference = np.abs(img - imgT)
        distance = np.sqrt(np.square(img) + np.square(imgT))  #Euclidean distance formula


        # TODO - Application 2 - Step 1c - add all pixels differences to a single number (score)
        score = np.sum(distance)



        # TODO - Application 2 - Step 1d - store the score and the label associated to imgT into the predictions list
        #  as a pair (score, label)
        predictions.append((score, y_train[idx][0])) # in each element of predictions we have a tuple (score, label) 



        #pass    #REMOVE THIS



    # TODO - Application 2 - Step 1e - Sort all elements in the predictions list in ascending order based on scores
    predictions = sorted(predictions, key=lambda x: x[0])  # sorting based on the score which is the first element in the tuple 



    # TODO - Application 2 - Step 1f - retain only the top k predictions
    k = 50
    top_predictions = predictions[0:k] #getting the first 10 elements from the sorted list




    # TODO - Application 2 - Step 1g - extract in a separate vector only the labels for the top k predictions
    predLabels = list(map(lambda x:x[1], top_predictions))




    # TODO - Application 2 - Step 1h - Determine the dominant class from the predicted labels
    predictedLabel = most_frequent(predLabels)




    return predictedLabel
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    # TODO - Application 1 - Step 1 - Load the CIFAR-10 dataset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

    # TODO - Exercise 1 - Determine the size of the four vectors x_train, y_train, x_test, y_test
    size_x_train = len(x_train)  
    size_y_train = len(y_train)
    size_x_test = len(x_test)
    size_y_test = len(y_test)
    print("Size of x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(size_x_train, size_y_train, size_x_test, size_y_test))

    print("Each image has a dimension of: {}".format(x_train[0].shape))

    # TODO - Exercise 2 - Visualize the first 10 images from the testing dataset with the associated labels

    for i in range(10):
        plt.imshow(x_test[i])
        plt.imsave('./ex'+str(i)+'.png', x_test[i])



    # TODO - Application 1 - Step 2 - Reshape the training and testing dataset from 32x32x3 to a vector
    x_train_flatten = np.float64(x_train.reshape((x_train.shape[0], 32 * 32 * 3))) # reshaping from a matrix to a vector of floats using the dimensions from before
    x_test_flatten = np.float64(x_test.reshape((x_test.shape[0], 32 * 32 * 3)))   


    numberOfCorrectPredictedImages = 0

    # TODO - Application 1 - Step 3 - Predict the labels for the first 100 images existent in the test dataset
    for idx, img in enumerate(x_test_flatten[0:200]):

        print("Make a prediction for image {}".format(idx))


        # TODO - Application 1 - Step 3 - Call the predictLabelNN function
        #predictedLabel = None  #Modify this
        #predictedLabel = predictLabelNN(x_train_flatten, y_train, img) #reshaped train set, labels of train set, current image from test set
        predictedLabel = predictLabelKNN(x_train_flatten, y_train, img)

        # TODO - Application 1 - Step 4 - Compare the predicted label with the groundtruth (the label from y_test).
        #  If there is a match then increment the contor numberOfCorrectPredictedImages
        if predictedLabel == y_test[idx][0]: ## comparing predicted label with actual label
            numberOfCorrectPredictedImages += 1




    # TODO - Application 1 - Step 5 - Compute the accuracy
    accuracy = (numberOfCorrectPredictedImages / 200.0) * 100 # total number of images predicted
    print("System accuracy = {}".format(accuracy))


    return
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################

