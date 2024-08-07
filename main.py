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


class OptimiserSGD:
    def __init__(self, learningRate=1, decay=0., momentum=0.):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weightMomentums'):
                layer.weightCache = np.zeros_like(layer.dweights)
                layer.biasCache = np.zeros_like(layer.dbiases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weightUpdates = self.momentum * layer.weightCache - self.currentLearningRate * layer.dweights
            layer.weightCache = weightUpdates
            # Build bias updates

            biasUpdates = self.momentum * layer.biasCache - self.currentLearningRate * layer.dbiases
            layer.biasCache = biasUpdates
        else:
            weightUpdates = - self.currentLearningRate * layer.dweights
            biasUpdates = - self.currentLearningRate * layer.dbiases

        layer.weights += weightUpdates
        layer.biases += biasUpdates

    def post_update_params(self):
        self.iterations += 1


class OptimiserAdagrad:
    def __init__(self, learningRate=1, decay=0., epsilon=1e-7):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightCache += layer.dweights ** 2
        layer.biasCache += layer.dbiases ** 2

        layer.weights += -self.currentLearningRate * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currentLearningRate * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimiserRMSprop:
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightCache = self.rho * layer.weightCache + (1 - self.rho) * layer.dweights ** 2
        layer.biasCache = self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.currentLearningRate * layer.dweights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currentLearningRate * layer.dbiases / (np.sqrt(layer.biasCache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class OptimiserAdam:
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # Initialize moment and cache vectors if not already done
        if not hasattr(layer, 'weightMomentums'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        # Update moment vectors with current gradients
        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dweights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 - self.beta1) * layer.dbiases

        # Compute bias-corrected moment estimates
        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dweights ** 2
        layer.biasCache = self.beta2 * layer.biasCache + (1 - self.beta2) * layer.dbiases ** 2

        # Compute bias-corrected cache estimates
        weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        biasCacheCorrected = layer.biasCache / (1 - self.beta2 ** (self.iterations + 1))

        # Update parameters
        layer.weights += (-self.currentLearningRate * weightMomentumsCorrected /
                          (np.sqrt(weightCacheCorrected) + self.epsilon))
        layer.biases += (-self.currentLearningRate * biasMomentumsCorrected /
                         (np.sqrt(biasCacheCorrected) + self.epsilon))

    def post_update_params(self):
        self.iterations += 1


if __name__ == '__main__':
    x, y = vertical_data(samples=100, classes=3)

    dense1 = LayerDense(2, 64)
    activation1 = ActivationRelu()

    dense2 = LayerDense(64, 3)

    lossActivation = ActivationSoftmaxLossCategoricalEntropy()
    optimiser = OptimiserAdam(learningRate=0.05, decay=1e-5)

    for i in range(10001):
        dense1.forward(x)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        loss = lossActivation.forward(dense2.output, y)

        predictions = np.argmax(lossActivation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not i % 100:
            print(
                f'epoch: {i} , ' + f'acc: {accuracy:.3f} , ' + f'loss: {loss:.3f} ' + f'lr: {optimiser.currentLearningRate:2f}')

        # backward pass
        lossActivation.backward(lossActivation.output, y)
        dense2.backward(lossActivation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimiser.pre_update_params()
        optimiser.update_params(dense1)
        optimiser.update_params(dense2)
        optimiser.post_update_params()
