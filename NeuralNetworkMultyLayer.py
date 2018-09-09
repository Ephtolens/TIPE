import numpy as np
from random import random, randint

class neuralNetwork:

	# Activations functions and their derivates

	sigmoid = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
	#dSigmoid = np.vectorize(lambda x : np.exp(-x)/(1+np.exp(-x))**2 )
	dSigmoid = np.vectorize(lambda x : x*(1-x))
	tanh = np.tanh
	dTanh = np.vectorize(lambda x : 1/np.cosh(x)**2)

	def __init__(self,inputs,hidden,outputs):
		self.nbInputs = inputs
		self.nbOutputs = outputs
		self.nbLayer = len(hidden)
		self.hiddenAr = hidden
		self.weightsAr = []
		self.biasAr = []

		for i in range(0,self.nbLayer + 1):
			if i == 0:
				a = self.hiddenAr[0]
				b = self.nbInputs
			elif i == self.nbLayer:
				a = self.nbOutputs
				b = self.hiddenAr[-1]
			else:
				a = self.hiddenAr[i]
				b = self.hiddenAr[i-1]	
			self.weightsAr.append(np.array([[(random() * 2 - 1) for _ in range(b)] for _ in range(a)]))
			self.biasAr.append([[(random() * 2 - 1)] for _ in range(a)])

			self.activation = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
			self.derivate = np.vectorize(lambda x : x*(1-x))

		self.lr = 0.01


	def setActivationFunction(self,function):

		if(function == 'sigmoid'):
			self.activation = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
			self.derivate = np.vectorize(lambda x : x*(1-x))
		elif(function == 'tanh'):
			self.activation = np.tanh
			self.derivate = np.vectorize(lambda x : 1 - (x * x))
 


	def predict(self,inputs_array,calcType = "predict"):
		inputs = np.transpose(np.array([inputs_array]))

		layersOutputAr = [np.array([]) for _ in range(self.nbLayer + 1)]

		for i in range(0,self.nbLayer+1):
			if i == 0:
				A = self.weightsAr[0]
				B = inputs
			else:
				A = self.weightsAr[i]
				B = layersOutputAr[i-1]
			bias = self.biasAr[i]

			layersOutputAr[i] = np.dot(A,B)
			layersOutputAr[i] += bias
			layersOutputAr[i] = self.activation(layersOutputAr[i])

		if calcType == "predict":
			return layersOutputAr[-1]
		elif calcType == "train":
			return layersOutputAr

	def train(self,inputs,target):

		layersOutputAr = self.predict(inputs,"train")
		

		inputs = np.transpose([inputs])
		target = np.transpose([target])

		errors = [np.array([]) for _ in range(self.nbLayer + 1)]
		errors[-1] = target - layersOutputAr[-1]

		for i in range(self.nbLayer,-1,-1):

			if (i != self.nbLayer):
				A = np.transpose(self.weightsAr[i+1])
				errors[i] = np.dot(A,errors[i+1])

			gradient = self.derivate(layersOutputAr[i])
			gradient *= errors[i]
			gradient *= self.lr

			if (i == 0):
				currentOutputT = np.transpose(inputs)
			else:
				currentOutputT = np.transpose(layersOutputAr[i-1])

			deltaWeight = np.dot(gradient,currentOutputT)

			self.weightsAr[i] += deltaWeight
			self.biasAr[i] += gradient

	def resetnetwork():
		self.weightsAr = []
		self.biasAr = []
		self.weightsAr.append(np.array([[(random() * 2 - 1) for _ in range(b)] for _ in range(a)]))
		self.biasAr.append([[(random() * 2 - 1)] for _ in range(a)])

				

#Test with xor
def xor():

	nn = neuralNetwork(2,[5,5],4)
	nn.setActivationFunction("tanh")

	trainingData = [[np.array([0,1]),np.array([1,0,0,0])],[np.array([1,0]),np.array([0,1,0,0])],[np.array([0,0]),np.array([0,0,1,0])],[np.array([1,1]),np.array([0,0,0,1])]]

	for i in range(50000):
		r = randint(0,3)
		nn.train(trainingData[r][0],trainingData[r][1])

	print(nn.predict(np.array([0,1])))
	print(nn.predict(np.array([1,0])))
	print(nn.predict(np.array([1,1])))
	print(nn.predict(np.array([0,0])))
