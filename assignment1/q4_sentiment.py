import numpy as np
import matplotlib.pyplot as plt

from cs224d.data_utils import *

from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper

import seaborn as sns
sns.set(style='whitegrid', context='talk')

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
REGULARIZATION = None   # Assign a list of floats in the block below
### YOUR CODE HERE
REGULARIZATION = np.logspace(-6,0.1,21)
REGULARIZATION = np.hstack([0,REGULARIZATION])

### END YOUR CODE

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the word vectors we trained earlier
_, wordVectors0, _ = load_saved_params()
N = wordVectors0.shape[0]//2
#assert nWords == N
wordVectors = (wordVectors0[:N,:] + wordVectors0[N:,:])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Try our regularization parameters
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization)

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels,
        weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)

    # Save the results and weights
    results.append({
        "reg" : regularization,
        "weights" : weights,
        "train" : trainAccuracy,
        "dev" : devAccuracy})

# Print the accuracies
print("")
print("=== Recap ===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%0.4g\t%0.4g" % (
        result["reg"],
        result["train"],
        result["dev"]))
for result in results:
    print("%0.2e & %0.4g & %0.4g \\\\" % (
        result["reg"],
        result["train"],
        result["dev"]))
print("")

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None

### YOUR CODE HERE
bestResult = max(results, key=lambda x: x["dev"])
BEST_REGULARIZATION = bestResult['reg']
BEST_WEIGHTS = bestResult['weight']

### END YOUR CODE

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value: %E" % BEST_REGULARIZATION)
print("Test accuracy (%%): %f" % accuracy(testLabels, pred))

# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper right')
plt.savefig("q4_reg_v_acc.png")
plt.show()
