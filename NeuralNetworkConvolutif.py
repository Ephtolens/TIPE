import numpy as np
from random import random,randint
from PIL import Image as im

sigmoid = np.vectorize(lambda x: (1/(1 + np.exp(-x))))
int  = np.vectorize(int)
imgOne = np.asarray(im.open("D:\\Média\\Photo\\Photo de profil\\Aout_2018.jpg"))
print(imgOne)

shapeim = np.shape(imgOne)
print(shapeim)
weight =  np.random.rand(shapeim[0],shapeim[1],shapeim[2])/1000
bias = np.random.rand(shapeim[0],shapeim[1],shapeim[2])/1000
pick = 4
def noyau_convolution(img, weightSet, biasSet, pickSize):
    '''Prend une 'image' en entrée de dimension 3, un set de Poids de meme dimension
    et un set de Biais de dimension 3 et retourne une 'image' de meme dimension que
    le set de Biais, tout les tableaux sont des tableaux Numpy'''

    forme_in = np.shape(img) #forme du tableau d'entrée
    forme_out = np.shape(biasSet) #forme du tableau de sortie
    output = np.zeros(forme_out) # Inisialisation du tableau de sortie à la bonne forme avec des zér0s

    for i in range(forme_in[0]-pickSize):
        for j in range(forme_in[1]-pickSize):
            S = 0
            for k in range(i,i+pickSize):
                for l in range(j,j+pickSize):
                    for c in range(forme_in[2]-1):
                        S += img[k,l,c]*weightSet[k,l,c]
            output[i,j,c] = S + biasSet[i,j,c]
    return (int(255*sigmoid(output)))

imgTwo = noyau_convolution(imgOne, weight, bias, pick)
imgTwo = np.array(imgTwo,dtype = "uint8")
imgTwo = im.fromarray(imgTwo)
imgTwo.show()
