import csv
import numpy as np

# Simple fully connected feedforward neural network with a single hidden layer
# Todo: add variable number of hidden layers
# Todo: add network edge specification instead of fully connected default

# Activation function
def activationFunc(x):
    return np.tanh(x)

# Derivative
def d_activationFunc(x):
    return 1 - (x ** 2)

class InputNeuron:

    def __init__(self, id):
        self.id = id
        self.value = 0.0
        self.outputs = []

    def getValue(self):
        return self.value

    def loadValue(self, value):
        self.value = value

    def addOutput(self, outputNeuron):
        self.outputs.append(outputNeuron)

class HiddenNeuron:

    def __init__(self, id):
        self.id = id
        # List of neurons from which we feed
        self.inputs = []
        # List of neurons wefeed (used later in backprop)
        self.outputs = []
        self.weights = np.array([])
        self.value = 0.0
        self.gradient = 0.0

    def updateValue(self):
        inputValues = []
        for i in self.inputs:
            inputValues.append(i.getValue())
        self.value = np.dot(np.concatenate(([1.0], inputValues)),self.weights)
        self.value = activationFunc(self.value)

    def getValue(self):
        return self.value
    
    def addOutput(self, outputNeuron):
        self.outputs.append(outputNeuron)

    def addInput(self, inputNeuron):
        self.inputs.append(inputNeuron)
        # Add ourselves on the input neuron as output
        inputNeuron.addOutput(self)
        # Resize weights array (+1 for bias)
        self.weights = 0.01 * np.random.rand(len(self.inputs)+1)

    def weightTo(self, neuron):
        # Return the weight to some input neuron
        i = 0
        for n in self.inputs:
            if n == neuron:
                return self.weights[i]
            i += 1
        # Should never end here
        raise RuntimeError("Invalid edge weight requested")

    def calcGradient(self):
        forwardError = 0.0
        for n in self.outputs:
            forwardError += n.gradient * n.weightTo(self)
        self.gradient = forwardError * d_activationFunc(self.value)

    def updateWeights(self, rate):
        for i in range(len(self.weights)):
            val = 1 if i == 0 else self.inputs[i-1].getValue()
            self.weights[i] += rate * val * self.gradient

class OutputNeuron(HiddenNeuron):

    def updateValue(self):
        inputValues = []
        for i in self.inputs:
            inputValues.append(i.getValue())
        self.value = activationFunc(np.dot(inputValues,self.weights))

    def addInput(self, inputNeuron):
        self.inputs.append(inputNeuron)
        # Add ourselves on the input neuron as output
        inputNeuron.addOutput(self)
        # Resize weights array
        self.weights = 0.1 * np.random.rand(len(self.inputs))


    def calcGradient(self, val):
        self.gradient = (val - self.value) * d_activationFunc(self.value)

    def updateWeights(self, rate):
        for i in range(len(self.weights)):
            self.weights[i] += rate * self.inputs[i].getValue() * self.gradient

class NeuralNet:

    # TODO: support more than 1 hidden layer, add connection specification
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, learningRate=1):
        
        # Learning rate
        self.learningRate = learningRate
        # Init neurons for each layer
        self.inputLayer = []
        self.hiddenLayer = []
        self.outputLayer = []

        for n in range(inputLayerSize):
            self.inputLayer.append(InputNeuron(n))

        for n in range(hiddenLayerSize):
            self.hiddenLayer.append(HiddenNeuron(n))

        for n in range(outputLayerSize):
            self.outputLayer.append(OutputNeuron(n))

        # Generate fully connected links from input to hidden layer
        for n in self.hiddenLayer:
            for i in self.inputLayer:
                n.addInput(i)

        # Generate fully connected links from hidden layer to output
        for n in self.outputLayer:
            for h in self.hiddenLayer:
                n.addInput(h)

    def forwardPass(self, inputArr):
        # Check input sizes match
        if len(inputArr) != len(self.inputLayer):
            raise ValueError("Input array size does not match inputLayer size")
        
        # Load input layer
        for n in self.inputLayer:
                n.loadValue(inputArr[n.id])

        # Trigger forward pass for the hidden layer
        for n in self.hiddenLayer:
            n.updateValue()

        res = []
        for n in self.outputLayer:
            n.updateValue()
            res.append(n.getValue())

        return res

    def backwardPass(self, outputArr):
        
        # First compute gradients for i-th output neurons
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].calcGradient(outputArr[i])

        # Then compute gradients in hidden layer
        for n in self.hiddenLayer:
            n.calcGradient()

        # Update weights on the output layer
        for n in self.outputLayer:
            n.updateWeights(self.learningRate)

        # Update weights on hidden layer
        for n in self.hiddenLayer:
            n.updateWeights(self.learningRate)

    # Fit a single data point
    def fit(self, inputArr, outputArr):
        # Compute forward pass result
        res = self.forwardPass(inputArr)

        # Check the training output size matches output layer size
        if len(outputArr) != len(res):
            raise RuntimeError("Training data label size does not match network output layer size")

        # Compute backward pass
        self.backwardPass(outputArr)

    # Batch fit
    def fitBatch(self, Xtrain, Ytrain, iters=100, errThreshold=0.01):

        print("Training using "+str(iters)+" iterations or until error is less than "+str(errThreshold))
        
        n = 0
        for n in range(iters):
            # Fit each data point
            for i in range(len(Xtrain)):
                nn.fit(Xtrain[i], Ytrain[i])

            # Print training accuracy
            err = 0.0
            for i in range(len(Xtrain)):
                res = nn.forwardPass(Xtrain[i])
                for j in range(len(res)):
                    err += 0.5 * ( (res[j] - Ytrain[i][j]) ** 2)
            err /= len(Xtrain)
            print("Epoch #" + str(n+1) + " avg err: " + str(err))
            if err < errThreshold: break

        print("Training done after epoch #"+str(n+1))

def loadData(file):
    X = []
    Y = []
    with open(file, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
        # Shuffle data
        np.random.shuffle(data)
        # First pass collect categorical vars
        YVals = []
        for row in data:
            last = row[len(row)-1]
            if last not in YVals:
                YVals.append(last)

        for row in data:
            X.append( [float (e) for e in row[:-1]] )
            Yval_num = YVals.index(row[-1])
            arr = [0] * len(YVals)
            arr[Yval_num] = 1
            Y.append(arr)

    return (X, Y)

def maxIndex(list):
    return list.index(max(list))

if __name__ == "__main__":
    
    # Load data
    XData, YData = loadData("iris.data")
    # Create net
    nn = NeuralNet(len(XData[0]), 10, len(YData[0]), 0.005)
    
    # Split data into train and test set 75-25
    splitAt = int(np.floor(0.5*len(XData)))
    
    XTrain = XData[:splitAt]
    YTrain = YData[:splitAt]
    
    XTest = XData[splitAt:]
    YTest = YData[splitAt:]
    
    nn.fitBatch(XTrain, YTrain, 100)

    # Test accuracy
    test_samples = len(XTest)
    test_hits = 0.0
    for i in range(test_samples):
        res = nn.forwardPass(XTest[i])
        resClass = maxIndex(res)
        if YTest[i][resClass] == 1.0:
            test_hits += 1.0

    print("Accuracy on test data: %"+str(test_hits*100.0/test_samples))
