import numpy as np


def predict(xsample, W):

    s = []
    # TODO - Application 3 - Step 2 - compute the vector with scores (s) as the product between W and xsample
    s = np.matmul(W, xsample) # matrix multiplication between W and the current input data point

    return s

# TODO - Application 3 - Step 3 - The function that compute the loss for a data point
def computeLossForASample(s, labelForSample, delta): # s -> score vector output from predict function, label -> true class label, delta -> margin

    loss_i = 0
    syi = s[labelForSample]  # the score for the correct class corresponding to the current input sample based on the label yi -> extracts the score for the true class that we are dealing now

    # TODO - Application 3 - Step 3 - compute the loss_i
    for j, sj in enumerate(s): # iterate through all the scores for all classes
        dist = sj - syi + delta

        if j == labelForSample: # skip the true class
            continue

        if dist > 0: # only add to loss if the margin condition is violated, it came from the formula
            loss_i += dist
        

    return loss_i

# TODO - Application 3 - Step 4 - The function that compute the gradient loss for a data point
# increases scores for incorrect classes and decreases score for correct class
def computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta):

    dW_i = np.zeros(W.shape)  # initialize the matrix of gradients with zero -> this will accumulate the gradients for the current sample
    syi = s[labelForSample]   # establish the score obtained for the true class 

    for j, sj in enumerate(s): # j is class index, sj is score for class j
        dist = sj - syi + delta

        if j == labelForSample: # doesn't check against itself
            continue

        if dist > 0:
            dW_i[j] = currentDataPoint # increase the weights for incorrect classes
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint # gets the gradient of the true class reduced, wich means the program saying " the true class score was too low, adjust the weights now to push it up"

    return dW_i

#####################################################################################################################
def main():

    # Input points in the 4 dimensional space
    x_train = np.array([[1, 5, 1, 4],
                        [2, 4, 0, 3],
                        [2, 1, 3, 3],
                        [2, 0, 4, 2],
                        [5, 1, 0, 2],
                        [4, 2, 1, 1]])

    # Labels associated with the input points
    y_train = [0, 0, 1, 1, 2, 2]

    # Input points for prediction
    x_test = np.array([[1, 5, 2, 4],
                       [2, 1, 2, 3],
                       [4, 1, 0, 1]])

    # Labels associated with the testing points
    y_test = [0, 1, 2]

    # The matrix of weights
    W = np.array([[-1, 2,  1, 3],
                  [ 2, 0, -1, 4],
                  [ 1, 3,  2, 1]])

    delta = 1               # margin
    step_size = 0.01        # weights adjustment ratio


    prev_loss = 1e9 
    threshold =  0.001 # threshold for stopping the training
    steps = 0
    max_steps = 1000

    while steps < max_steps:
        loss_L = 0
        dW = np.zeros(W.shape) # class with the same shape initialized with zeros
    
        # TODO - Application 3 - Step 2 - For each input data...
        for idx, xsample in enumerate(x_train):

            # TODO - Application 3 - Step 2 - ...compute the scores s for all classes (call the method predict)
            s = predict(xsample, W) # scores for all classes for the current input data point 



            # TODO - Application 3 - Step 3 - Call the function (computeLossForASample) that
            #  compute the loss for a data point (loss_i)
            loss_i = computeLossForASample(s, y_train[idx], delta)



            # Print the scores - Uncomment this
            print("Scores for sample {} with label {} is: {} and loss is {}".format(idx, y_train[idx], s, loss_i))



            # TODO - Application 3 - Step 4 - Call the function (computeLossGradientForASample) that
            #  compute the gradient loss for a data point (dW_i)
            dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)



            # TODO - Application 3 - Step 5 - Compute the global loss for all the samples (loss_L)
            loss_L += loss_i # accumulate the loss for all samples



            # TODO - Application 3 - Step 6 - Compute the global gradient loss matrix (dW)
            dW += dW_i # accumulate the gradient for all samples

        loss_L = loss_L / len(x_train)  # normalize the loss by the number of training samples
        dw = dW / len(x_train) # normalize the gradient by the number of training samples
        loss_variation = abs(loss_L - prev_loss)

        if loss_variation < threshold: # stopping condition based on loss variation
            print("Training stopped after {} steps.".format(steps))
            break

        W = W - step_size * dW  # update the weights by moving in the direction of negative gradient
        prev_loss = loss_L
        steps += 1
        print(f"Converged after {steps} steps with loss {loss_L}")






    # TODO - Application 3 - Step 7 - Compute the global normalized loss
    loss_L = loss_L / len(x_train)  # normalize the loss by the number of training samples
    print("The global normalized loss = {}".format(loss_L))



    # TODO - Application 3 - Step 8 - Compute the global normalized gradient loss matrix
    dW = dW / len(x_train) # normalize the gradient by the number of training samples



    # TODO - Application 3 - Step 9 - Adjust the weights matrix
    W = W - step_size * dW  # update the weights by moving in the direction of negative gradient




    # TODO - Exercise 7 - After solving exercise 6, predict the labels for the points existent in x_test variable
    #  and compare them with the ground truth labels. What is the system accuracy?
    correctPredicted = 0
    for idx, xsample in enumerate(x_test):



        pass    # REMOVE THIS

    accuracy = 0   # Modify this
    print("Accuracy for test = {}%".format(accuracy))

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
