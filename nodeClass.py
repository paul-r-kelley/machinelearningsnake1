#Returns the dot product of two lists
def dotProduct(list1, list2):
    total = 0
    
    for i in range(len(list1)):
        total += list1[i] * list2[i]
    return total

#The maximum values of bias and weight, all values will be between 0 and 1
maxBias = 1
maxWeight = 1


#Takes inputs from layer above in the network, and produces an output according to its weights and bias.
class Node:
    def __init__(self, numberOfNodesInAboveLayer):
        self.bias = 0
        self.weights = []
        self.numberOfNodesInAboveLayer = numberOfNodesInAboveLayer

    #Returns the output value for this particular node, which becomes the input to nodes in the next layer.
    def output(self, nodesInAboveLayer):
        nodeOutputs = []

        for item in nodesInAboveLayer:
            nodeOutputs.append(item)
        return dotProduct(deepcopy(nodeOutputs), self.weights) + self.bias

    #Randomizes the bias and all the weights for this node.
    def randomize(self):
        self.bias = random.random() * maxBias  
        newWeights = []

        for i in range(self.numberOfNodesInAboveLayer):
            newWeights.append(random.random() * maxWeight)

        self.weights = newWeights

    #Return the bias value for this node
    def getBias(self):
        return self.bias
    
    #Return the list of weights for this node
    def getWeights(self):
        return self.weights

    #Set the bias value for this node to the passed value
    def setBias(self, newBias):
       self.bias = newBias

    #Set the weights for this node to the passed values
    def setWeights(self, newWeights):
        self.weights = deepcopy(newWeights)
