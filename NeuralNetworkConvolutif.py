import numpy as np
from random import random,randint

L = 700
l = 700
p = 3
weight = np.array([[None]*2100]*700) #Poids du pixel [i,j,k] = weight[i,j+k*l]
for i in range (700):
    for j in range(2100):
        weight[i,j] = random()

