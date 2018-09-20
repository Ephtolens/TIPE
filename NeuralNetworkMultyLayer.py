import numpy as np
from random import random, randint, gauss
import dill as d
# Module pour le puissance 4
class neuralNetwork:
	"""
		Classe permettant la création de réseaux de neuronnes entièrement connectés

		Lors de la création, le constructeur a besoin d'un entier pour le nombre d'entrées du réseau, d'un tableau d'entier contenant dans
		chaque case le nombre de neuronnes sur la couche du numéro de la case et d'un entier représentant le nombre de sorties souhaité


	"""


	def __init__(self,inputs = 1,hidden = [1],outputs = 1):
		""" Initialisation du réseau / int, int array, int"""

		if type(inputs) == int:

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
				self.weightsAr.append(np.array([[random() for _ in range(b)] for _ in range(a)]))
				self.biasAr.append([[random()] for _ in range(a)])

		else:
			self.nbInputs = inputs.nbInputs
			self.nbOutputs = inputs.nbOutputs
			self.nbLayer = inputs.nbLayer
			self.hiddenAr = inputs.hiddenAr
			self.weightsAr = inputs.weightsAr
			self.biasAr = inputs.biasAr

		self.activation = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
		self.derivate = np.vectorize(lambda x : x*(1-x))

		self.lr = 0.01



	def setActivationFunction(self,function):
		""" Méthode changeant la fonction d'activation du réseau / string: sigmoid ou tanh"""

		if(function == 'sigmoid'):
			self.activation = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
			self.derivate = np.vectorize(lambda x : x*(1-x))
		elif(function == 'tanh'):
			self.activation = np.tanh
			self.derivate = np.vectorize(lambda x : 1 - (x * x))
 


	def predict(self,inputs_array,calcType = "predict"):
		""" Méthode de prédiction d'un résultat / array des valeurs d'entrée"""

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
		""" Méthode d'entrainement du réseau / array des valeurs d'entrée, array des valeurs de sortie souhaitées """

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

	def reset(self):
		""" Méthode réinitialisant les matrices du réseau / None """

		answer = input("Voulez vous enregistrer votre réseau avant de l'écraser? oui/non ").lower()

		if answer == "oui":
			fileName = input("Veuillez saisir le nom de fichier: ")
			self.save(fileName)

		self.__init__(self.nbInputs,self.hiddenAr,self.nbOutputs)


	def save(self,fileName):
		""" Méthode enregistrant le réseau / string , string pour le nom du réseau et le nom de fichier"""

		title = "Neural Network ({})".format(fileName)	# Définition du nom du fichier
		d.dump(self,open(title,"wb"))	# Enregistrement du réseau via le module dill
		print("Réseau enregistré sous le nom: " + title)

	def load(self,fileName):
		""" Méthode chargeant un réseau enregistré depuis cette librairie / string pour le nom du fichier à charger"""

		answer = input("Voulez vous enregistrer votre réseau avant de l'écraser? oui/non ").lower()

		if answer == "oui":
			fileName = input("Veuillez saisir le nom de fichier: ")
			self.saveNetwork(fileName)

		title = "Neural Network ({})".format(fileName)	# Définition du nom du fichier
		A = d.load(open(title,"rb"))	# Chargement du réseau via le module dill

		self.__init__(A.nbInputs,A.hiddenAr,A.nbOutputs)
		self.weightsAr = A.weightsAr
		self.biasAr = A.biasAr			
		self.lr = A.lr 					# Affectation du réseau chargé dans le réseau actuel
		self.activation = A.activation
		self.derivate = A.derivate

		print("Réseau chargé depuis le fichier: " + title)

	def mutate(self,number):
		# Mute le réseau, change un certain nombre de weights

		for l in range(len(self.weightsAr)):

			for i in range(len(self.weightsAr[l])):
				for j in range(len(self.weightsAr[l][i])):
					if random() <= number:
						self.weightsAr[l][i][j] += gauss(0, 0.1)

		for l in range(len(self.biasAr)):

			for i in range(len(self.biasAr[l])):
				for j in range(len(self.biasAr[l][i])):
					if random() <= number:
						self.biasAr[l][i][j] += gauss(0, 0.1)

	def normalize(self,output):

		S = 0

		for i in output:
			S += i

		for i in range(len(output)):
			output[i] /= S

		return output


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
