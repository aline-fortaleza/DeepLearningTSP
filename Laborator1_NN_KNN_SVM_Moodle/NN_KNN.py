import numpy as np
import cv2
import tensorflow as tf
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
dict_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def most_frequent(list):
    list = [x[0] for x in list]
    return [max(set(list), key = list.count)]
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 1 - Step 3 - Compute the difference between the current image (img) taken from test dataset
#  with all the images from the train dataset. Return the label for the training image for which is obtained
#  the lowest score
def predictLabelNN(x_train_flatten, y_train, img):

    predictedLabel = -1
    scoreMin = 100000000

    # TODO - Application 1 - Step 3a - for each image in the training list
    for idx, imgT in enumerate(x_train_flatten):

        # TODO - Application 1 - Step 3b - compute the absolute difference between img and imgT
        # difference = ...


        # TODO - Application 1 - Step 3c - add all pixels differences to a single number (score)
        # score = ...


        # TODO - Application 1 - Step 3d - retain the label where the minimum score is obtained




        pass   #REMOVE THIS

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
        #difference = ...



        # TODO - Application 2 - Step 1c - add all pixels differences to a single number (score)
        #score = ...



        # TODO - Application 2 - Step 1d - store the score and the label associated to imgT into the predictions list
        #  as a pair (score, label)



        pass    #REMOVE THIS



    # TODO - Application 2 - Step 1e - Sort all elements in the predictions list in ascending order based on scores
    #predictions = ...



    # TODO - Application 2 - Step 1f - retain only the top k predictions




    # TODO - Application 2 - Step 1g - extract in a separate vector only the labels for the top k predictions
    predLabels = []




    # TODO - Application 2 - Step 1h - Determine the dominant class from the predicted labels
    #predictedLabel = ...




    return predictedLabel
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    # TODO - Application 1 - Step 1 - Load the CIFAR-10 dataset



    # TODO - Exercise 1 - Determine the size of the four vectors x_train, y_train, x_test, y_test



    # TODO - Exercise 2 - Visualize the first 10 images from the testing dataset with the associated labels



    # TODO - Application 1 - Step 2 - Reshape the training and testing dataset from 32x32x3 to a vector
    x_train_flatten = []  #Modify this
    x_test_flatten = []   #Modify this


    numberOfCorrectPredictedImages = 0

    # TODO - Application 1 - Step 3 - Predict the labels for the first 100 images existent in the test dataset
    for idx, img in enumerate(x_test_flatten[0:200]):

        print("Make a prediction for image {}".format(idx))


        # TODO - Application 1 - Step 3 - Call the predictLabelNN function
        predictedLabel = None  #Modify this


        # TODO - Application 1 - Step 4 - Compare the predicted label with the groundtruth (the label from y_test).
        #  If there is a match then increment the contor numberOfCorrectPredictedImages




    # TODO - Application 1 - Step 5 - Compute the accuracy
    accuracy = 0  #Modify this
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

