#Controls the generation and testing of networks.
class Controller:
    #This constructor takes in the number of generations to train with, and the structure of the network. Ex: [10, 20, 4] would be 10 and 20 nodes in the hidden layer, and 4 in the output layer.
    def __init__(self,numberOfGenerations, nodesPerLayer):
        self.generation = []
        self.nodesPerLayer = nodesPerLayer
        self.numberOfGenerations = numberOfGenerations
        self.networksPerGeneration = 100
        self.bestNetworks = []
        self.bestFitnesses = []
        self.fitnesses = []
        self.games = []
        self.currentNetworkID = 0

        for i in range(numberOfGenerations):
            print("Starting new generation!")
            self.generation = []

            if i == 0:
                #If first generation, just create 100 random networks
                for j in range(self.networksPerGeneration):
                    self.generation.append(Network(nodesPerLayer))
                    self.generation[j].createRandNetwork()
                    self.generation[j].setID(self.currentNetworkID)
                    self.currentNetworkID += 1
            else:
                #If not first generation, weightedmutate the top 5 networks of last generation with each other, then generate 3 random mutations from each of those, giving 100 total.
                for j in range(len(self.bestNetworks[i-1])):
                    commonNetwork = copy(self.bestNetworks[i-1][j])
                    commonFitness = copy(self.bestFitnesses[i-1][j])

                    for k in range(len(self.bestNetworks[i-1])):
                        if j != k:
                            currentNetwork = copy(self.bestNetworks[i-1][k])
                            currentFitness = copy(self.bestFitnesses[i-1][k])

                            newNetwork = copy(weightedMutate(copy(commonNetwork), copy(currentNetwork), commonFitness, currentFitness))
                            newNetwork.setID(self.currentNetworkID)
                            self.currentNetworkID += 1
                            self.generation.append(copy(newNetwork))
                        else:
                            self.generation.append(copy(commonNetwork))

                #3 random mutates on each element of self.generation[i]
                for j in range(25):
                    mutate1 = copy(randomMutate(deepcopy(self.generation[j])))
                    mutate1.setID(self.currentNetworkID)
                    self.currentNetworkID += 1            
                    self.generation.append(deepcopy(mutate1))

                    mutate2 = deepcopy(randomMutate(deepcopy(self.generation[j])))
                    mutate2.setID(self.currentNetworkID)
                    self.currentNetworkID += 1
                    self.generation.append(deepcopy(mutate2))

                    mutate3 = deepcopy(randomMutate(deepcopy(self.generation[j])))
                    mutate3.setID(self.currentNetworkID)
                    self.currentNetworkID += 1
                    self.generation.append(deepcopy(mutate3))

            print("Networks generated!")

            generationFitnesses = []
            genFitnessesNoMod = []
            generationLengths = []

            for j in range(len(self.generation)):
                #All 100 networks play snake, storing their fitness.
                fitnessScore, lengthScore = self.networkFitness(copy(self.generation[j]))

                generationFitnesses.append(fitnessScore)
                genFitnessesNoMod.append(fitnessScore)
                generationLengths.append(lengthScore)

            #Find top 5 fitness scores, save those networks to bestNetworks.
            print("BestIndex: {}, FitScore: {}".format(generationFitnesses.index(max(generationFitnesses)), max(generationFitnesses)))
            bestForGeneration = []
            bestFitnessesForGeneration = []

            for j in range(5):
                maxIndex = generationFitnesses.index(max(generationFitnesses))
                bestForGeneration.append(self.generation.pop(maxIndex))
                bestFitnessesForGeneration.append(generationFitnesses.pop(maxIndex))

            print("MaxFitness for generation {}: {}".format(i, max(genFitnessesNoMod)))
            print("AvgFitness for generation {}: {}".format(i, sum(genFitnessesNoMod) / len(genFitnessesNoMod)))
            print("MaxLength for generation {}: {}".format(i, max(generationLengths)))
            print("AvgLength for generation {}: {}".format(i, sum(generationLengths) / len(generationLengths)))
            self.bestNetworks.append(deepcopy(bestForGeneration))
            self.fitnesses.append(deepcopy(generationFitnesses))
            self.bestFitnesses.append(bestFitnessesForGeneration)

            #Accounts for the fact that the 5 best from the previous generation continue on into the next, so this keeps the naming conventions constant
            self.currentNetworkID += 5


    #Takes a current network and runs the game with it, returning the overall fitness score.
    def networkFitness(self, network):
        game = Game()
        game.play(network)

        finalMoveFitness = game.getTotalMoveFitness()
        finalLength = game.getFinalLength()
        stepCount = game.getStepCount()
        self.games.append(game.getGameStates())

        stepCountWeight = 0.001
        moveFitnessWeight = 5
        lengthWeight = 20

        #Calculate the final fitness for this network, finalLength is subtracted by 1 because the snake's head is included in that length.
        finalFitness = (finalMoveFitness * moveFitnessWeight) + ((finalLength - 1)* lengthWeight) + (stepCount * stepCountWeight)
        if finalFitness < 0:
            finalFitness = 0
        return finalFitness, finalLength
    
    #Return the 3D array representing any particular game just played.
    def getGame(self, index):
        return self.games[index]
