import numpy as np
from pathlib import Path



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

def accuracy_on_set(X, y, W):
    correctPredicted = 0
    for idx in range(len(X)):
        s = predict(X[idx], W)  # compute scores for all classes
        y_pred = np.argmax(s)  # predicted label is the index of the class with the highest score

        if y_pred == y[idx]:
            correctPredicted += 1
    return (correctPredicted / len(X))

def load_iris_csv(path="Iris.csv"): 
    path = Path(__file__).resolve().parent / path

    data = np.genfromtxt(path, delimiter=",", dtype=str)

    #if first row is header (non-numeric), drop it
    try:
        float(data[0, 0])
    except ValueError:
        data = data[1:, :]

    X = data[:, 0:4].astype(float)
    y_raw = data[:, 4]

    #map labels to 0,1,2 (stable ordering by sorted unique names)
    classes = sorted(np.unique(y_raw))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[v] for v in y_raw], dtype=int)

    return X, y, classes


def train_one_run(X_train, y_train, X_test, y_test, step_size, delta=1.0,
                  max_steps=1000, threshold=1e-3, seed=0):
    rng = np.random.default_rng(seed)

    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    #W random in [0,1] as requested
    W = rng.random((num_classes, num_features))

    prev_loss = 1e9
    first_step_over_90 = None

    for step in range(max_steps):
        loss_L = 0.0
        dW = np.zeros_like(W)

        for i in range(len(X_train)):
            s = predict(X_train[i], W)
            loss_i = computeLossForASample(s, y_train[i], delta)
            dW_i = computeLossGradientForASample(W, s, X_train[i], y_train[i], delta)

            loss_L += loss_i
            dW += dW_i

        loss_L /= len(X_train)
        dW /= len(X_train)

        # monitor test accuracy (this is what the exercise asks)
        test_acc = accuracy_on_set(X_test, y_test, W)
        if first_step_over_90 is None and test_acc >= 0.90:
            first_step_over_90 = step

        loss_variation = abs(loss_L - prev_loss)
        if loss_variation < threshold:
            break

        W = W - step_size * dW
        prev_loss = loss_L

    final_test_acc = accuracy_on_set(X_test, y_test, W)
    return final_test_acc, first_step_over_90


def main():
    X, y, classes = load_iris_csv("Iris.csv")

    #shuffle + split (120 train / 30 test)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:], y[120:]

    #standardization of features 
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-12
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    #step sizes to try 
    step_sizes = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    seeds = [0, 1, 2, 3, 4]  #test random initialization influence

    best = None  # (mean_acc, step_size, details)

    print("Classes:", classes)
    print("Running sweep...\n")

    for lr in step_sizes: # learning rate
        accs = []
        steps90 = []

        for sd in seeds: # random seed
            acc, first90 = train_one_run(X_train, y_train, X_test, y_test,
                                         step_size=lr, delta=1.0,
                                         max_steps=1000, threshold=1e-3, seed=sd)
            accs.append(acc)
            steps90.append(first90 if first90 is not None else np.inf)

        mean_acc = float(np.mean(accs))
        min_steps_90 = int(np.min(steps90)) if np.min(steps90) != np.inf else None
        any_100 = any(a == 1.0 for a in accs)

        print(f"step_size={lr:<7} | mean test acc={mean_acc:.3f} | min steps to >=90%={min_steps_90} | any 100%? {any_100}")

        if best is None or mean_acc > best[0]:
            best = (mean_acc, lr, accs, steps90)

    mean_acc, best_lr, accs, steps90 = best
    min_steps_90 = int(np.min([s for s in steps90 if s != np.inf])) if any(s != np.inf for s in steps90) else None


    print(f"Best step_size (highest mean test accuracy): {best_lr}")
    print(f"Mean test accuracy with best step_size: {mean_acc:.3f}")
    print(f"Minimum steps to reach >=90% (best step_size, across seeds): {min_steps_90}")
    print(f"Accuracies across seeds (best step_size): {[round(a*100,1) for a in accs]}")

if __name__ == "__main__":
    main()