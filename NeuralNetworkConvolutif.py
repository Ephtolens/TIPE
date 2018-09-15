import numpy as np
import neuralNetwork as nn
from random import random,randint

pictPath = ""
pictWidth = 10
pictHeight = 10
pickSize = 4


imgOne = # NP array de l'image

# *** Première convolution ***
imgTwo = np.array([[0 for _ in range(pictWidth)] for i in range (pictHeight)])# NP array vide pour la première convolution
rankOneW = np.array([[np.array([random(),random(),random()]) for _ in range(pictWidth)] for i in range (pictHeight)]) #Poids des pixels de imgOne
rankOneB = np.array([[random() for _ in range(pictWidth)] for i in range (pictHeight)]) # Biais pour la première convolution

for i in range(pictHeight-pickSize):
    for j in range(pictWidth-pickSize):
        S = 0
        for k in range(i,i+pickSize):
            for l in range(j,j+pickSize):
                S += imgOne[k,l,0]*rankOneW[k,l,0] + imgOne[k,l,1]*rankOneW[k,l,1] + imgOne[k,l,2]*rankOneW[k,l,2]
        imgTwo[i,j] = S + rankOneB[i,j]
nn.sigmoid(imgTwo)


 # *** Deuxieme convolution ***
imgThree = np.array([0 for _ in range((pictHeight-2*pickSize)*(pictWidth-2*pickSize))] #Np array vide pour la deuxieme convolution
rankTwoW = np.array([[np.array([random(),random(),random()]) for _ in range(pictWidth)] for i in range (pictHeight)]) #Poids des pixels de imgTwo
rankTwoB = np.array([[random() for _ in range(pictWidth)] for i in range (pictHeight)]) # Biais pour la deuxieme convolution

height = pictHeight-2*pickSize
width = pictWidth-2*pickSize
for i in range(height):
    for j in range(width):
        S = 0
        for k in range(i,i+pickSize):
            for l in range(j,j+pickSize):
                S += imgTwo[k,l]*rankTwoW[k,l]
        imgThree[i + width * j] = S + rankTwoB[i,j]

nn.sigmoid(imgThree)

def convolution(img, weightSet, biasSet, pickSize):
    '''Prend une 'image' en entrée, les pois et biais associés et retourne une 'image' de dim n (déduite de biasSet)
