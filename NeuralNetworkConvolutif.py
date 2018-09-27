import numpy as np
import NeuralNetworkMultyLayer as nn
from random import random,randint
from PIL import Image as im

pictPath = ""
pictWidth = 10
pictHeight = 10
pickSize = 4


imgOne = np.asarray(im.open("D:\\Média\\Photo\\Photo de profil\\Aout_2018.jpg"))
print(imgOne)
imgOne = im.fromarray(imgOne)
imgOne.show()

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
imgThree = np.array([0 for _ in range((pictHeight-2*pickSize)*(pictWidth-2*pickSize))]) #Np array vide pour la deuxieme convolution
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
    '''Prend une 'image' en entrée, les pois et biais associés et retourne une 'image' de dim n (déduite de biasSet)'''

    forme = np.shape(img) #forme : (hauteur,largeur,profondeur)

    output = np.zeros(forme)

    for i in range(forme[0]-pickSize):
        for j in range(forme[1]-pickSize):
            S = 0
            for k in range(i,i+pickSize):
                for l in range(j,j+pickSize):
                    if (forme[2]):
                        for c in range(forme[2]):
                            S += img[k,l,c]*weightSet[k,l,c]
                    else :
                        S += img[k,l]*weightSet[k,l]
            output[i,j,c] = S + biasSet[i,j]
