import nnfs
import numpy as np

from nnfs.datasets import vertical_data


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Loss:
    def calculate(self, output, y):
        # Calculates the data and regularization losses
        # given model output and ground truth values
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss


class ActivationSoftmax:
    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = expValues / np.sum(expValues, axis=1, keepdims=True)


class LossCategoricalCrossentropy(Loss):
    def forward(self, yPred, yTrue):
        # number of samples in batch
        samples = len(yPred)

        # Clipping data to prevent division by 0
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            correctConfidences = np.sum(yPredClipped * yTrue, axis=1)

        negativeLogLikelihoods = -np.log(correctConfidences)
        return negativeLogLikelihoods


if __name__ == '__main__':

    x, y = vertical_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()

    lossFunction = LossCategoricalCrossentropy()

    #   helper variables
    lowestLoss = 999999
    bestDense1Weights = dense1.weights.copy()
    bestDense1Biases = dense1.biases.copy()
    bestDense2Weights = dense2.weights.copy()
    bestDense2Biases = dense2.biases.copy()

    for iteration in range(10000):
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        dense1.forward(x)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = lossFunction.calculate(activation2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if loss < lowestLoss:
            print("new weights found,iteration:", iteration, 'loss: ', loss, 'acc: ', accuracy)
            bestDense1Weights = dense1.weights.copy()
            bestDense1Biases = dense1.biases.copy()
            bestDense2Weights = dense2.weights.copy()
            bestDense2Biases = dense2.biases.copy()
            lowestLoss = loss
        else:
            dense1.weights = bestDense1Weights.copy()
            dense1.biases = bestDense1Biases.copy()
            dense2.weights = bestDense2Weights.copy()
            dense2.biases = bestDense2Biases.copy()
