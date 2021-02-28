#List of nodes, their weights, and biases. Takes an input and produces an output by passing its input through all the nodes.
class Network:
    def __init__(self, nodesPerLayer):
        self.id = -1
        #nodesPerLayer is a list of integers, representing how many nodes are in each layer (excluding input)
        self.nodesPerLayer = nodesPerLayer
        self.nodes = []
        self.boardState = []


    #Creates a network where the weights and biases for every node are random. Used only for the 1st generation.
    def createRandNetwork(self):
        for i in range(len(self.nodesPerLayer)):

            nodesInCurrentLayer = []
            for j in range(self.nodesPerLayer[i]):
                if i == 0:
                    nodesInCurrentLayer.append(Node(2))
                else:
                    nodesInCurrentLayer.append(Node(deepcopy(self.nodesPerLayer[i - 1])))
            self.nodes.append(nodesInCurrentLayer)

        for i in range(len(self.nodesPerLayer)):
            for item in self.nodes[i]:
                item.randomize()


    #Returns the direction that the neural network thinks the snake should travel in.
    def predict(self, boardState):
        self.boardState = deepcopy(boardState)
        networkOutputs = []
        flatBoard = []
                   
        #Need to turn board from 9x9 into 1D array of 81 length
        """
        for i in range(9):
            for j in range(9):
                #print("i: {}, j: {}".format(i, j))
                flatBoard.append(self.boardState[i][j])
        """

        xDist, yDist, headX, headY = getDistanceComponents(boardState)
        
        if headY + 1 > 8:
            below = -100
        else:
            below = boardState[headX][headY + 1] * 10

        if headY - 1 < 0:
            above = -100
        else:
            above = boardState[headX][headY - 1] * 10

        if headX - 1 < 0:
            left = -100
        else:
            left = boardState[headX - 1][headY] * 10

        if headX + 1 > 8:
            right = -100
        else:
            right = boardState[headX + 1][headY] * 10

        self.inputData = []
        self.inputData.append(xDist)
        self.inputData.append(yDist)
        #self.inputData.append(above)
        #self.inputData.append(below)
        #self.inputData.append(left)
        #self.inputData.append(right)

        for i in range(len(self.nodes)):
            outputsForThisLayer = []

            for j in range(len(self.nodes[i])):
                if i == 0:
                    #Layer above is the input layer
                    outputsForThisLayer.append(self.nodes[i][j].output(deepcopy(self.inputData)))             
                else:
                    #Standard behavior
                    outputsForThisLayer.append(self.nodes[i][j].output(deepcopy(networkOutputs[i - 1])))

            networkOutputs.append(deepcopy(outputsForThisLayer))
        maxOutput = max(networkOutputs[-1])

        #The direction which the network has decided to go
        maxIndex = networkOutputs[-1].index(maxOutput)
        return maxIndex


    #Prints the network id, as well as the weights and bias for every node in the network
    def print(self):
        print("NetworkID : {}".format(self.id))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                print("Node[{}][{}]: Bias = {}".format(i, j, self.nodes[i][j].getBias()))
                weights = deepcopy(self.nodes[i][j].getWeights())
                for k in range(len(weights)):
                    print("Index: {}, Weight: {}".format(k, weights[k]))

    #Sets the network id equal to the passed value
    def setID(self, newID):
        self.id = newID

    #Returns the network's id
    def getID(self):
        return self.id


#Takes in two networks and returns a new network which is the average  of every value across all nodes weighted according to the network's fitnesses
def weightedMutate(network1, network2, network1Fitness, network2Fitness):
    network3 = Network(network1.nodesPerLayer)
    network3.createRandNetwork()

    totalFitness = network1Fitness + network2Fitness

    if totalFitness == 0:
        network1FitFactor = 0.5
        network2FitFactor = 0.5
    else:      
        network1FitFactor = network1Fitness / totalFitness
        network2FitFactor = network2Fitness / totalFitness
    
    for i in range(len(network1.nodes)):
        for j in range(len(network1.nodes[i])):

            net1Node = network1.nodes[i][j]
            net2Node = network2.nodes[i][j]

            #Set weights and biases of new network to be the average of the the weights and biases by node of the two networks, taking fitness into account.
            newBias = ((net1Node.getBias() * network1FitFactor) + (net2Node.getBias() * network2FitFactor))
            network3.nodes[i][j].setBias(newBias)
            
            net1NodeWeights = deepcopy(net1Node.getWeights())
            net2NodeWeights = deepcopy(net2Node.getWeights())
            newWeights = []

            for k in range(len(net1NodeWeights)):
                newWeights.append(((net1NodeWeights[k] * network1FitFactor) + (net2NodeWeights[k] * network2FitFactor)))

            network3.nodes[i][j].setWeights(deepcopy(newWeights))

    return network3


#Picks a random element of a network and randomizes it to a new value betweenn 0 and 1 up to numberOfMutations times.
def randomMutate(network):
    numberOfMutations = 5
    newNetwork = network
    nodesPerLayer = network.nodesPerLayer

    for i in range(numberOfMutations):
        randLayer = random.randint(0, len(nodesPerLayer) - 1)
        randNode = random.randint(0, nodesPerLayer[randLayer] - 1)

        currentNode = newNetwork.nodes[randLayer][randNode]

        thingsToChange = []
        thingsToChange.append(currentNode.getBias())

        weights = deepcopy(currentNode.getWeights())
        for item in weights:
            thingsToChange.append(item)

        randomFeature = random.randint(0, len(thingsToChange) - 1)
        thingsToChange[randomFeature] = random.random()

        newNetwork.nodes[randLayer][randNode].setBias(thingsToChange.pop(0))
        
        for k in range(len(thingsToChange)):
            newNetwork.nodes[randLayer][randNode].weights[k] = thingsToChange[k]

    return newNetwork


#Returns the x and y distance from the snake's head to the fruit, as well as the x and y position of the snake's head.
def getDistanceComponents(boardState):
    fruitX = -1
    fruitY = -1
    headX = -1
    headY = -1
    tempX = 0
    tempY = 0

    while fruitX == -1 or headX == -1:
        if boardState[tempY][tempX] == fruit:
            fruitX = tempX
            fruitY = tempY
        elif boardState[tempY][tempX] == snakeHead:
            headX = tempX
            headY = tempY

        if tempX == 8:
            tempY += 1
            tempX = 0
        else:
            tempX += 1

    beforeMoveDX = headX - fruitX
    beforeMoveDY = headY - fruitY
    return beforeMoveDX, beforeMoveDY, headX, headY

#Returns the fitness score for the last move made.
def getMoveFitnessScore(boardState, direction):
    fruitX = -1
    fruitY = -1
    headX = -1
    headY = -1
    tempX = 0
    tempY = 0

    while fruitX == -1 or headX == -1:
        if boardState[tempY][tempX] == fruit:
            fruitX = tempX
            fruitY = tempY
        elif boardState[tempY][tempX] == snakeHead:
            headX = tempX
            headY = tempY

        if tempX == 8:
            tempY += 1
            tempX = 0
        else:
            tempX += 1

    #Compare distance from snakehead to fruit before and after move
    beforeMoveDX = pow(headX - fruitX, 2)
    beforeMoveDY = pow(headY - fruitY, 2)
    beforeMoveDist = math.sqrt(beforeMoveDX + beforeMoveDY)

    postMoveHeadX = headX
    postMoveHeadY = headY

    if direction == 0:
        #Left
        postMoveHeadX -= 1
    elif direction == 1:
        #Down
        postMoveHeadY += 1
    elif direction == 2:
        #Right
        postMoveHeadX += 1
    elif direction == 3:
        #Up
        postMoveHeadY -= 1
    else:
        print("Invalid direction in getMoveFitnessScore.")

    postMoveDX = pow(postMoveHeadX - fruitX, 2)
    postMoveDY = pow(postMoveHeadY - fruitY, 2)
    postMoveDist = math.sqrt(postMoveDX + postMoveDY)

    #If positive, snake is now closer to fruit, else snake is further from fruit
    changeInDist = beforeMoveDist - postMoveDist

    """
    if changeInDist < 0:
        changeInDist = 0
    """

    return changeInDist
