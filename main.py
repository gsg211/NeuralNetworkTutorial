import nnfs
import numpy as np

from nnfs.datasets import vertical_data


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # calculating gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationRelu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


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

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            singleOutput = singleOutput.reshape(-1, 1)
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)


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

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)

        # number of labels in each sample
        labels = len(dvalues[0])
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        # gradient
        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalEntropy:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, yTrue):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, yTrue)

    def backward(self, dvalues, yTrue):
        samples = len(dvalues)

        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), yTrue] -= 1
        self.dinputs = self.dinputs / samples


if __name__ == '__main__':
    x, y = vertical_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)

    lossActivation = ActivationSoftmaxLossCategoricalEntropy()

    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = lossActivation.forward(dense2.output, y)
    print(lossActivation.output[: 5])
    print('loss:', loss)

    predictions = np.argmax(lossActivation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    print("acc: ", accuracy)

    # backward pass
    lossActivation.backward(lossActivation.output, y)
    dense2.backward(lossActivation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Print gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)
