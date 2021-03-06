# -*- coding: utf-8 -*-
"""Project3 - Snake Game.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cUQrE9HIieTP42cWcyU3JD5s66dybTmz
"""
import random
from random import randint
import math
from copy import copy, deepcopy

#Constants for how the board is stored.
blank = 0
snakeHead = 2
snakeBody = 1
fruit = 5

#A hard limit for all games
maxNumberOfMoves = 500

#The class which contains all the logic for the Snake game.
class Game:
    def __init__(self):
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 2, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.snakeSegments = []
        self.snakeLength = 1
        self.alive = True
        self.headPos = [-1, -1]
        self.totalMoveFitness = 0
        self.stepCount = 0
        self.fruitPositions = []
        self.gameStates = []

    #Prints the full 2D array representing the game board.
    def printBoard(self):
        for i in range(9):
            print(self.board[i])

    #Performs all the setup and variable resetting needed before starting a new game.
    def setup(self):
        self.resetBoard()
        self.headPos[0] = 4
        self.headPos[1] = 4
        self.alive = True
        self.totalMoveFitness = 0
        self.snakeLength = 1
        self.snakeSegments = []
        self.stepCount = 0
        self.fruitPositions = [[3, 3], [7, 4], [1, 7], [7, 7], [3, 5], [6, 2], [3, 6], [8, 2], [5, 5], [0, 8]]
        self.gameStates = []        
        self.placeRandomFruit()
        self.gameStates.append(self.board)

    #Returns the length of the snake at time of death
    def getFinalLength(self):
        return self.snakeLength

    #Returns the total fruitDistance fitness across all moves
    def getTotalMoveFitness(self):
        return self.totalMoveFitness

    #Returns the total number of steps taken at time of death
    def getStepCount(self):
        return self.stepCount

    #Returns the 3D array storing the board after every move
    def getGameStates(self):
        return self.gameStates

    #Resets the board to a new state (before placing a fruit)
    def resetBoard(self):
        for i in range(9):
            for j in range(9):
                self.board[i][j] = blank
        
        self.board[4][4] = snakeHead

    #Sets alive variable to False, called when game ends.
    def end(self):
        self.alive = False

    #Places a fruit anywhere on the board that the snake isn't
    def placeRandomFruit(self):
        fruitX = random.randint(0, 8)
        fruitY = random.randint(0, 8)

        while self.board[fruitX][fruitY] != blank:
            fruitX = random.randint(0, 8)
            fruitY = random.randint(0, 8)

        self.board[fruitX][fruitY] = fruit

    #Places a fruit on the board according to a pre-selected list of positions, but switches to random after 10 fruits are placed.
    def placeConstantFruit(self):
        if self.snakeLength <= 10:
            fruitPos = deepcopy(self.fruitPositions[self.snakeLength - 1])
            fruitX = fruitPos[0]
            fruitY = fruitPos[1]
        else:              
            fruitX = random.randint(0, 8)
            fruitY = random.randint(0, 8)

        while self.board[fruitX][fruitY] != blank:
            if self.snakeLength <= 10:
                fruitPos = deepcopy(self.fruitPositions[self.snakeLength - 1])
                fruitX = fruitPos[0]
                fruitY = fruitPos[1]
            else:              
                fruitX = random.randint(0, 8)
                fruitY = random.randint(0, 8)

        self.board[fruitX][fruitY] = fruit

    #Takes in the direction of travel and moves the snake and its segments
    def updateSnakePos(self, newX, newY, fruitCollected):
        if fruitCollected:
            self.snakeLength += 1                   
        
        self.board[newX][newY] = snakeHead
        if self.snakeLength > 1:
            if fruitCollected:
                self.board[self.headPos[0]][self.headPos[1]] = snakeBody
                self.snakeSegments.insert(0, [self.headPos[0], self.headPos[1]])
                self.placeRandomFruit()
            else:
                self.board[self.headPos[0]][self.headPos[1]] = snakeBody
                self.snakeSegments.insert(0, [self.headPos[0], self.headPos[1]])
                oldTail = self.snakeSegments.pop()        
                self.board[oldTail[0]][oldTail[1]] = blank          
        else:
            self.board[self.headPos[0]][self.headPos[1]] = blank
        
        self.headPos[0] = newX
        self.headPos[1] = newY

        self.gameStates.append(deepcopy(self.board))
        

    #Handles the collisions for moving the snake in a particular direction
    def move(self, direction):
        self.stepCount += 1

        if direction == 0:
            #Left
            newX = self.headPos[0]
            newY = self.headPos[1] - 1
        elif direction == 1:
            #Down
            newX = self.headPos[0] + 1
            newY = self.headPos[1]
        elif direction == 2:
            #Right
            newX = self.headPos[0]
            newY = self.headPos[1] + 1
        elif direction == 3:
            #Up
            newX = self.headPos[0] - 1
            newY = self.headPos[1]
        else:
            print("Invalid direction chosen!")

        self.totalMoveFitness += getMoveFitnessScore(self.board, direction)
        
        if newX in range(0, 9) and newY in range(0, 9):
            space = self.board[newX][newY]
            if space == blank:
                self.updateSnakePos(newX, newY, False)
            elif space == snakeBody:
                #Snake collided with its tail
                self.end()
            elif space == fruit:
                self.updateSnakePos(newX, newY, True)
            else:
                print("Invalid tile status. Tile was {}".format(space))
        else:
            #Snake collided with wall of the board.
            self.end()
    
    
    #Starts a new game
    def play(self, network):
        self.setup()
        while self.alive:

            if self.stepCount > maxNumberOfMoves:
                self.end()

            direction = network.predict(deepcopy(self.board))
            self.move(direction)
